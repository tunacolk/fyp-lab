"""Two-stage recommender: candidate generation -> learned ranker -> MMR diversity re-rank.
Plus a cold-start path for users with no history.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _sparse(pairs, n_users, n_items):
    return sp.csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_users, n_items))


def _zrow(M):
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)


class RecSys:
    def __init__(self, ds, n_cand=60, mf_dim=32, text_dim=24):
        self.ds, self.n_cand, self.mf_dim, self.text_dim = ds, n_cand, mf_dim, text_dim

    # ------------------------------------------------------------------ fit
    def fit(self):
        ds = self.ds
        nu, ni = ds.n_users, ds.n_items
        self.R = _sparse(ds.base, nu, ni)                                  # history used for features
        self.seen = (self.R + _sparse(ds.val, nu, ni)).tocsr()             # base + val: never re-recommend
        self.pop = np.asarray(self.R.sum(0)).ravel()

        # 1) collaborative filtering: truncated SVD on the implicit matrix
        svd = TruncatedSVD(self.mf_dim, random_state=0).fit(self.R)
        self.mf = svd.transform(self.R) @ svd.components_

        # 2) content-based: TF-IDF + SVD on item titles (swap in Sentence-Transformers in prod)
        tfidf = TfidfVectorizer().fit_transform(ds.items["title"])
        self.E = normalize(TruncatedSVD(self.text_dim, random_state=0).fit_transform(tfidf))
        prof = normalize(self.R @ self.E)
        self.content = prof @ self.E.T

        # user->category affinity and activity
        onehot = sp.csr_matrix((np.ones(ni), (np.arange(ni), ds.item_cat)),
                               shape=(ni, ds.item_cat.max() + 1))
        share = (self.R @ onehot).toarray()
        self.cat_share = share / (share.sum(1, keepdims=True) + 1e-9)
        self.act = np.log1p(np.asarray(self.R.sum(1)).ravel())

        self.mf_z, self.content_z = _zrow(self.mf), _zrow(self.content)
        self._fit_ranker()
        return self

    # ------------------------------------------------------- candidate stage
    def _mask_seen(self, M, users, train=False):
        """train=True masks only base history (so val items can appear as ranker positives)."""
        M = M.copy()
        rows, cols = (self.R if train else self.seen)[users].nonzero()
        M[rows, cols] = -np.inf
        return M

    @staticmethod
    def _topk(M, k):
        idx = np.argpartition(-M, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(M, idx, 1), axis=1)
        return np.take_along_axis(idx, order, 1)

    def candidates(self, users, train=False):
        n = self.n_cand
        mf = self._topk(self._mask_seen(self.mf[users], users, train), n)
        ct = self._topk(self._mask_seen(self.content[users], users, train), n)
        pop = self._topk(self._mask_seen(np.tile(self.pop, (len(users), 1)).astype(float), users, train), n)
        return [np.unique(np.concatenate([a, b, c])) for a, b, c in zip(mf, ct, pop)]

    # ----------------------------------------------------------- ranker stage
    def _features(self, u, items):
        return np.column_stack([
            self.mf[u, items], self.content[u, items],
            self.mf_z[u, items], self.content_z[u, items],
            np.log1p(self.pop[items]),
            self.cat_share[u, self.ds.item_cat[items]],
            np.full(len(items), self.act[u]),
        ])

    def _fit_ranker(self):
        ds = self.ds
        users = ds.warm_users
        val_by_user = {}
        for u, i in ds.val:
            val_by_user.setdefault(u, set()).add(i)
        X, y = [], []
        for u, cands in zip(users, self.candidates(users, train=True)):
            X.append(self._features(u, cands))
            y.append(np.isin(cands, list(val_by_user.get(u, ()))).astype(int))
        self.ranker = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_depth=4, random_state=0
        ).fit(np.vstack(X), np.concatenate(y))

    # -------------------------------------------------------------- serving
    def recommend_scores(self, users, kind="mf", k=10):
        """kind in {pop, mf, content}: single-model baselines."""
        if kind == "pop":
            M = np.tile(self.pop, (len(users), 1)).astype(float)
        else:
            M = getattr(self, kind)[users]
        top = self._topk(self._mask_seen(M, users), k)
        return {int(u): list(map(int, r)) for u, r in zip(users, top)}

    def recommend_ranked(self, users, k=10, mmr_lambda=None, pool=40):
        out = {}
        for u, cands in zip(users, self.candidates(users)):
            p = self.ranker.predict_proba(self._features(u, cands))[:, 1]
            order = np.argsort(-p)
            if mmr_lambda is None:
                out[int(u)] = list(map(int, cands[order[:k]]))
            else:
                out[int(u)] = self._mmr(cands[order[:pool]], p[order[:pool]], k, mmr_lambda)
        return out

    def _mmr(self, cands, rel, k, lam):
        """Maximal Marginal Relevance: trade relevance against similarity to already-picked items."""
        rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
        chosen, rest = [], list(range(len(cands)))
        while len(chosen) < k and rest:
            if not chosen:
                best = rest[int(np.argmax(rel[rest]))]
            else:
                sim = self.E[cands[rest]] @ self.E[cands[chosen]].T
                score = lam * rel[rest] - (1 - lam) * sim.max(1)
                best = rest[int(np.argmax(score))]
            chosen.append(best); rest.remove(best)
        return [int(cands[i]) for i in chosen]

    # ------------------------------------------------------------ cold start
    def recommend_cold(self, onboarding_cats, k=10, explore=2, use_onboarding=True):
        """No history: onboarding categories (popularity inside them) + a few exploration slots."""
        gpop = np.argsort(-self.pop)
        if not use_onboarding:
            return list(map(int, gpop[:k]))
        picked = []
        per = [[i for i in gpop if self.ds.item_cat[i] == c] for c in onboarding_cats]
        j = 0
        while len(picked) < k - explore and any(j < len(p) for p in per):
            for p in per:
                if j < len(p) and len(picked) < k - explore:
                    picked.append(int(p[j]))
            j += 1
        for i in gpop:                                   # exploration: globally popular, other categories
            if len(picked) >= k:
                break
            if int(i) not in picked:
                picked.append(int(i))
        return picked
