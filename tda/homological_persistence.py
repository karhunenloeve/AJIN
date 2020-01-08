import matplotlib.pyplot as plt
import numpy as np


palette = ['#ff0000', '#00ff00', '#0000ff', '#00ffff', '#ff00ff', '#ffff00',
           '#000000', '#880000', '#008800', '#000088', '#888800', '#880088',
           '#008888']


def __min_birth_max_death(persistence):
    """This function returns (min_birth, max_death) from the persistence.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :returns: (float, float) -- (min_birth, max_death).
    """
    # Look for minimum birth date and maximum death date for plot optimisation
    max_death = 0
    min_birth = persistence[0][1][0]
    for interval in reversed(persistence):
        if float(interval[1][1]) != float('inf'):
            if float(interval[1][1]) > max_death:
                max_death = float(interval[1][1])
        if float(interval[1][0]) > max_death:
            max_death = float(interval[1][0])
        if float(interval[1][0]) < min_birth:
            min_birth = float(interval[1][0])
    return (min_birth, max_death)

 
def show_palette_values(alpha=0.6):
    """This function shows palette color values in function of the dimension.

    :param alpha: alpha value in [0.0, 1.0] for horizontal bars (default is 0.6).
    :type alpha: float.
    :returns: plot -- An horizontal bar plot of dimensions color.
    """
    colors = []
    for color in palette:
        colors.append(color)

    y_pos = np.arange(len(palette))

    plt.barh(y_pos, y_pos + 1, align='center', alpha=alpha, color=colors)
    plt.ylabel('Dimension')
    plt.title('Dimension palette values')

    plt.show()


def plot_persistence_barcode(persistence, alpha=0.6):
    """This function plots the persistence bar code.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param alpha: alpha value in [0.0, 1.0] for horizontal bars (default is 0.6).
    :type alpha: float.
    :returns: plot -- An horizontal bar plot of persistence.
    """
    (min_birth, max_death) = __min_birth_max_death(persistence)
    ind = 0
    delta = ((max_death - min_birth) / 10.0)
    # Replace infinity values with max_death + delta for bar code to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta

    # Draw horizontal bars in loop
    for interval in reversed(persistence):
        if float(interval[1][1]) != float('inf'):

            # Finite death case
            plt.barh(ind, (interval[1][1] - interval[1][0]), height=0.8,
                     left = interval[1][0], alpha=alpha,
                     color = palette[interval[0]])
        else:

            # Infinite death case for diagram to be nicer
            plt.barh(ind, (infinity - interval[1][0]), height=0.8,
                     left = interval[1][0], alpha=alpha,
                     color = palette[interval[0]])

        ind = ind + 1

    plt.title('Persistence barcode')

    # Ends plot on infinity value and starts a little bit before min_birth
    plt.axis([axis_start, infinity, 0, ind])
    plt.show()


def plot_persistence_diagram(persistence, alpha=0.6):
    """This function plots the persistence diagram.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param alpha: alpha value in [0.0, 1.0] for points and horizontal infinity line (default is 0.6).
    :type alpha: float.
    :returns: plot -- An diagram plot of persistence.
    """
    (min_birth, max_death) = __min_birth_max_death(persistence)
    ind = 0
    delta = ((max_death - min_birth) / 10.0)
    # Replace infinity values with max_death + delta for diagram to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta

    # line display of equation : birth = death
    x = np.linspace(axis_start, infinity, 1000)
    # infinity line and text
    plt.plot(x, x, color='k', linewidth=1.0)
    plt.plot(x, [infinity] * len(x), linewidth=1.0, color='k', alpha=alpha)
    plt.text(axis_start, infinity, r'$\infty$', color='k', alpha=alpha)

    # Draw points in loop
    for interval in reversed(persistence):
        if float(interval[1][1]) != float('inf'):
            # Finite death case
            plt.scatter(interval[1][0], interval[1][1], alpha=alpha,
                        color = palette[interval[0]])
        else:
            # Infinite death case for diagram to be nicer
            plt.scatter(interval[1][0], infinity, alpha=alpha,
                        color = palette[interval[0]])
        ind = ind + 1

    plt.title('Persistence diagram')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    # Ends plot on infinity value and starts a little bit before min_birth
    plt.axis([axis_start, infinity, axis_start, infinity + delta])
    return(plt)


# persistence_diagram and bootstrap
def plot_persistence_diagram_boot(persistence, alpha=0.6,band_boot=0):
    """This function plots the persistence diagram with confidence band 

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param alpha: alpha value in [0.0, 1.0] for points and horizontal infinity line (default is 0.6).
    :param band_boot: bootstrap band 
    :type alpha: float.
    :returns: plot -- An diagram plot of persistence.
    """
    (min_birth, max_death) = __min_birth_max_death(persistence)
    ind = 0
    delta = ((max_death - min_birth) / 10.0)
    # Replace infinity values with max_death + delta for diagram to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta

    # line display of equation : birth = death
    x = np.linspace(axis_start, infinity, 1000)
    # infinity line and text
    plt.plot(x, x, color='k', linewidth=1.0)
    plt.plot(x, [infinity] * len(x), linewidth=1.0, color='k', alpha=alpha)
    # bootstrap band 
    plt.fill_between(x, x, x+band_boot, alpha = 0.3, facecolor='red')
    #,alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99',linewidth=0)   

    plt.text(axis_start, infinity, r'$\infty$', color='k', alpha=alpha)

    # Draw points in loop
    for interval in reversed(persistence):
        if float(interval[1][1]) != float('inf'):
            # Finite death case
            plt.scatter(interval[1][0], interval[1][1], alpha=alpha,
                        color = palette[interval[0]])
        else:
            # Infinite death case for diagram to be nicer
            plt.scatter(interval[1][0], infinity, alpha=alpha,
                        color = palette[interval[0]])
        ind = ind + 1

    plt.title('Persistence diagram')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    # Ends plot on infinity value and starts a little bit before min_birth
    plt.axis([axis_start, infinity, axis_start, infinity + delta])
    plt.show()


def landscapes_approx(diag_dim,x_min,x_max,nb_steps,nb_landscapes):
    landscape = np.zeros((nb_landscapes,nb_steps))
    step = (x_max - x_min) / nb_steps
    #Warning: naive and not the best way to proceed!!!!!
    for i in range(nb_steps):
        x = x_min + i * step
        event_list = []
        for pair in diag_dim:
            b = pair[0]
            d = pair[1]
            if (b <= x) and (x<= d):
                if x >= (d+b)/2. :
                    event_list.append((d-x))
                else:
                    event_list.append((x-b))
        event_list.sort(reverse=True)
        event_list = np.asarray(event_list)
        for j in range(nb_landscapes):
            if(j<len(event_list)):
                landscape[j,i]=event_list[j]
    return landscape


def truncated_simplex_tree(st,int_trunc=100):
    """This function return a truncated simplex tree  
    :st : a simplex tree
    :int_trunc : number of persistent interval keept per dimension (the largest) 
    """
    st.persistence()    
    dim = st.dimension()
    st_trunc_pers = [];
    for d in range(dim):
        pers_d = st.persistence_intervals_in_dimension(d)
        d_l= len(pers_d)
        if d_l > int_trunc:
            pers_d_trunc = [pers_d[i] for i in range(d_l-int_trunc,d_l)]
        else:
            pers_d_trunc = pers_d
        st_trunc_pers = st_trunc_pers + [(d,(l[0],l[1])) for l in pers_d_trunc]
    return(st_trunc_pers)


