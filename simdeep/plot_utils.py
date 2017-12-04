from colour import Color
import numpy as np


CSS = """
table
{
  border-collapse: collapse;
}
th
{
  color: #ffffff;
  background-color: #000000;
}
td
{
  background-color: #cccccc;
}
table, th, td
{
  font-family:Arial, Helvetica, sans-serif;
  border: 1px solid black;
  text-align: right;
}
"""


class SampleHTML():
    def __init__(self, name, label, proba, survival):
        """
        """
        nbdays, isdead = survival

        self.html =  """
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>{0}</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Assigned class</th>
      <td>{1}</td>
    </tr>
    <tr>
      <th>class probability</th>
      <td>{2}</td>
    </tr>
    <tr>
      <th>nb days followed</th>
      <td>{3}</td>
    </tr>
    <tr>
      <th>is dead</th>
      <td>{4}</td>
    </tr>
  </tbody>
</table>
                """.format(name, label, proba, nbdays, isdead)


def make_color_dict_from_r(labels):
    """ """
    labels_set = set(labels)

    cin = Color('red')
    cout = Color('#56f442')

    gradient = map(lambda x:x.get_rgb(),
                   cin.range_to(cout, len(labels_set)))

    len_color = len(gradient)

    if len_color > 2:
        gradient[1] = Color('green').get_rgb()
        gradient[2] = Color('blue').get_rgb()

    if len_color > 3:
        gradient[3] = Color('cyan').get_rgb()

    if len_color > 4:
        gradient[4] = Color('magenta').get_rgb()

    if len_color > 5:
        gradient[5] = Color('yellow').get_rgb()

    return dict(zip(labels_set, gradient))


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
