import tests.test_all as t, traceback
ok = 0
fns = [f for n, f in vars(t).items() if n.startswith("test_")]
for f in fns:
    try: f(); ok += 1; print("PASS", f.__name__)
    except Exception: print("FAIL", f.__name__); traceback.print_exc()
print(f"{ok}/{len(fns)} passed")
