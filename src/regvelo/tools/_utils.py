
def split_elements(character_list):
    """Split elements."""
    result_list = []
    for element in character_list:
        if '_' in element:
            parts = element.split('_')
            result_list.append(parts)
        else:
            result_list.append([element])
    return result_list

def combine_elements(split_list):
    """Combine elements."""
    result_list = []
    for parts in split_list:
        combined_element = "_".join(parts)
        result_list.append(combined_element)
    return result_list

def get_list_name(lst):
    """Get the names of the elements in a dictionary."""
    names = []
    for name, obj in lst.items():
        names.append(name)
    return names



