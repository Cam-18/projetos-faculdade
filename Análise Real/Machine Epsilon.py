def find_eps():
    z = 1.0
    while 1.0 + z > 1.0:
        z /= 2.0
    return z * 2.0

eps = find_eps()
print(f'eps = {eps}')
