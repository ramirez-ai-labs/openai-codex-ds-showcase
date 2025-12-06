
def is_palindrome(s):
    s_clean = ''.join(ch.lower() for ch in s if not ch.isspace())
    return s_clean == s_clean[::-1]
