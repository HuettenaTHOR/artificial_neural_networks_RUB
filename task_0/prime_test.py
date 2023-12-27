def prime_test(num):
    if num < 2:
        return False
    for i in range(2, int(num / 2) + 2):
        if num % i == 0:
            return False
    return True


if __name__ == "__main__":
    print(prime_test(2))
    print(prime_test(9))
    print(prime_test(31))
