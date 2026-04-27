
def words2characters(words):
    """
    This function converts a list of words into a list of characters.

    @param:
    words - a list of words

    @return:
    characters - a list of characters

    Every element of "words" should be converted to a str, then split into
    characters, each of which is separately appended to "characters." For 
    example, if words==['hello', 1.234, True], then characters should be
    ['h', 'e', 'l', 'l', 'o', '1', '.', '2', '3', '4', 'T', 'r', 'u', 'e']
    """
    characters = []
    for item in words:
        item_str = str(item)
        for char in item_str:
            characters.append(char)
    return characters


def cancellation(input_list, stop_word):

    output_list = []
    for item in input_list:
        if item == stop_word:
            break
        else:
            output_list.append(item)
    return output_list


def copy_all_but_skip_word(input_list, skip_word):

    output_list = []
    for item in input_list:
        if item == skip_word:
            continue
        else:
            output_list.append(item)
    return output_list


def my_average(input_list):

    return sum(input_list) / len(input_list)