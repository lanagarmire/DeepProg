from colour import Color
import numpy as np


def make_color_list(id_list):
    """
    According to an id_list define a color gradient
    return {id:color}
    """
    try:
        assert([Color(idc) for idc in id_list])
    except Exception:
        pass
    else:
        return id_list

    color_dict = make_color_dict(id_list)

    return np.array([color_dict[label] for label in id_list])

def make_color_dict(id_list):
    """
    According to an id_list define a color gradient
    return {id:color}
    """
    id_list = list(set(id_list))

    first_c = Color("red")
    middle_c = Color("green")

    m_length1 = len(id_list)

    gradient = list(first_c.range_to(middle_c, m_length1))

    color_dict =  {id_list[i]: gradient[i].get_hex_l()
                   for i in range(len(id_list))}

    return color_dict
