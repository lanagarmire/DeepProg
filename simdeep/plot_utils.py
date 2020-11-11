from sklearn.decomposition import PCA

from colour import Color
import numpy as np

import matplotlib
matplotlib.use('Agg')

import seaborn as sns

import pylab as plt
import mpld3

sns.set(color_codes=True)


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
        try:
            nbdays, isdead = survival
        except Exception:
            nbdays, isdead = 'NaN', 'NaN'

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
      <th>Event</th>
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

    gradient = list(map(lambda x:x.get_rgb(),
                   cin.range_to(cout, len(labels_set))))

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

def plot_kernel_plots(
        test_labels,
        test_labels_proba,
        labels,
        activities,
        activities_test,
        dataset,
        path_html,
        metadata_frame=None):
    """
    perform a html kernel plot
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    color_dict = make_color_dict_from_r(labels)
    labels_c_test = np.array([color_dict[label] for label in test_labels])

    decomp = PCA(n_components=2)
    X, Y = decomp.fit_transform(activities).T

    X_test, Y_test = decomp.transform(activities_test).T

    for label in set(labels):
        ax.scatter(
            X_test[test_labels == label],
            Y_test[test_labels == label],
            s=40,
            # linewidths=2.0,
            alpha=1.0,
           # marker='square_cross',
            edgecolors='k',
            zorder=2,
           color=labels_c_test[test_labels == label],
           label='test cluster nb {0}'.format(label))

        sns.kdeplot(
            X[labels == label],
            Y[labels == label],
            shade=True,
            cmap=sns.dark_palette(color_dict[label], as_cmap=True),
            color=color_dict[label],
            ax=ax,
            label='cluster nb {0}'.format(label),
            zorder=1,
            thresh=False,
            alpha=0.7
        )

    survival_test = np.nan_to_num(dataset.survival_test)

    labels = [SampleHTML(
        name=dataset.sample_ids_test[i],
        label=test_labels[i],
        survival=np.asarray(survival_test[i])[0],
        proba=test_labels_proba[i][test_labels[i]]).html
              for i in range(len(test_labels))]

    scatter = ax.plot(X_test, Y_test, 'o', color='b', mec='k',
                      ms=15, mew=1, alpha=0.0, zorder=3,)[0]

    tooltip = mpld3.plugins.PointHTMLTooltip(
        scatter, labels, voffset=10, hoffset=10, css=CSS)
    mpld3.plugins.connect(fig, tooltip)

    mpld3.save_html(fig, path_html)

    print('kde plot saved at:{0}'.format(path_html))
