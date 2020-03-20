import string

from genetic import encrypt, decrypt, get_coding_map


def test_encrypt():
    message = "I like cats"

    coding_map = get_coding_map()

    encrypted_message = encrypt(coding_map, message)

    reversed_coding_map = {v: k for k, v in coding_map.items()}

    assert message.lower() == decrypt(reversed_coding_map, encrypted_message)
