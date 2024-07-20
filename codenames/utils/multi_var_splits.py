import itertools


EDUCATION = [
    "EDUCATION high school associate",
    "EDUCATION bachelor",
    "EDUCATION graduate",
]

RELIGION = [
    "RELIGION catholic",
    "RELIGION not catholic",
]

POLITICAL = [
    "POLITICAL liberal",
    "POLITICAL conservative",
]

COUNTRY = [
    "COUNTRY united states",
    "COUNTRY foreign",
]

def get_all_two_splits():
    result = []
    cultural_vars = [EDUCATION, RELIGION, POLITICAL, COUNTRY]
    for var1, var2 in itertools.combinations(cultural_vars, 2):
        pairs = []
        for p in itertools.product(var1, var2):
            pairs.append(sorted(p))
        result.extend(pairs)
    return result

if __name__ == "__main__":
    print(get_all_two_splits())