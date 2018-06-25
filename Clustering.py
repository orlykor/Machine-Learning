import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import euclidean_distances
import random
import sklearn.manifold.t_sne as tsne
from sklearn import datasets

def circles_example():
    """
    an example function for generating and plotting synthesised data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)


    # plt.plot(circles[0,:], circles[1,:], '.k')
    # plt.show()
    return circles


def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    # plt.plot(apml[:, 0], apml[:, 1], '.')
    # plt.show()
    return apml


def microarray_exploration(data_path='microarray_data.pickle',
                            genes_path='microarray_genes.pickle',
                            conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5,5],[-5,5],'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    return euclidean_distances(X, Y)


def euclidean_centroid(C_k):
    """
    return the center of mass of data points of C_k.
    :param C_k: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """

    return np.sum(C_k, axis=0)/C_k.shape[0]



def kmeans_pp_init(X, K, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param K: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """

    N = X.shape[0]
    d = X.shape[1]
    random_index = random.randint(0, N-1)

    centroids = np.zeros((K, d))
    centroids[0] = X[random_index]

    for k in range (1, K):
        # the d is in the size of N and the current amount of cetroids
        d = metric(X, centroids)
        D = np.amin(d, 1).reshape(N, 1)
        W = np.square(D)
        vec_prob = W / np.sum(W)
        centroids[k] = _choose_random(X, N, vec_prob)
    return centroids


def _choose_random(X, N, prob):
    index = np.random.choice(N, p=prob[:,0])
    return X[index]

def silhouette(X, clustering, centroid):
    """
    Given results from clustering with K-means, return the silhouette measure of
    the clustering.
    :param X: The NxD data matrix.
    :param clustering: N-dimensional vector, representing the
                clustering with the minimal score of the iterations of K-means.
    :param centroid: kxD centroid matrix
    :return: The Silhouette statistic, for k selection.
    """

    k = centroid.shape[0]
    N = X.shape[0]
    a = np.zeros(N)
    all_distances = np.zeros((N, k))
    for i in range(k):
        indices = np.where(clustering == i)[0]
        C_k = X[indices]
        a[indices] = np.sum(euclid(C_k, C_k), 0)/C_k.shape[0]
        all_distances[:, i] = np.sum(euclid(X, C_k), 1)/C_k.shape[0]
    all_distances[np.arange(N), clustering] = np.nan
    b = np.nanmin(all_distances, 1)
    S = np.sum((b-a)/np.maximum(b, a))
    return S




def run_n_times_kmeans(X, k, iterations=1, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init, stat=silhouette):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :param stat: A function for calculating the statistics we want to extract about
                the result (for K selection, for example).
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    statistics - whatever data you choose to use for your statistics (silhouette by default).
    """

    clustering = [None] * iterations
    centroids = [None] * iterations
    score = [None] * iterations

    for i in range(iterations):
        clustering[i], centroids[i] = _kmeans(X, k, metric, center, init)
        score[i] = _score_kmeans(X, clustering[i], centroids[i], k, metric)

    opt_index = np.argmin(np.array(score))
    cost = score[opt_index]
    clustering = clustering[opt_index]
    centroids = centroids[opt_index]
    statistics = stat(X, clustering, centroids)

    return clustering, centroids, statistics, cost

def _score_kmeans(X, clustering, centroids, k, metric):

    score = 0
    for i in range(k):
        C_k = X[np.where(clustering == i)[0]]
        score += np.sum(metric(C_k, centroids[i, np.newaxis]) ** 2)

    return score

def _kmeans(X, k, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):

    centroids = init(X, k, metric)
    clustering = None
    converges = False
    while not converges:
        dist = metric(X, centroids)
        bef_centroids = np.copy(centroids)
        clustering = np.argmin(dist, 1)
        for i in range(k):
            C_k = X[np.where(clustering == i)[0]]
            centroids[i] = center(C_k)
        if np.allclose(bef_centroids, centroids):
            converges = True
    return clustering, centroids

def heat(S, sigma):
    """
    calculate the heat kernel similarity of the given data matrix.
    :param S: A NxD distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """

    W = np.exp(-(S ** 2)/(2*(sigma**2)))
    return W

def mnn(S, m):
    """
    calculate the m nearest neighbors similarity of the given data matrix.
    :param S: A NxD distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    mnn_matrix = np.zeros(S.shape)
    neighbors = np.argsort(S, 1)[:, :m]
    for i in range(m):
        mnn_matrix[np.arange(S.shape[0]), neighbors[:, i]] = 1
    return mnn_matrix


def spectral(X, k, similarity_param, similarity=heat, is_dim_reduction=False):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the hear kernel.
    :param similarity: The similarity transformation of the data.
    :param is_dim_reduction: made to determine if the spectral is used for
    dimension reduction
    :return: all what the kmeans returns.
    """

    S = euclidean_distances(X)
    W = similarity(S, similarity_param)
    minus_squart_D = np.power(np.sum(W, 0), -0.5)
    D = np.diag(minus_squart_D)
    L = np.identity(D.shape[0]) - np.dot(D, W).dot(D)
    eigen_vecs = np.linalg.eigh(L)[1]
    data = np.matrix(eigen_vecs[:, 0:k])

    if is_dim_reduction:
        return data

    return run_n_times_kmeans(data, k)


def _run_examples(data, k, iterations, title):

    clustering = [None] * k
    centroids = [None] * k
    statistics = [None] * k
    costs = [None] * k

    for i in range(k):
        clustering[i], centroids[i], statistics[i], costs[i] = \
            run_n_times_kmeans(data, i+2, iterations)

    plt.figure()
    plt.plot(np.arange(2, k+2), statistics)
    silhouette_title = "Silhouette as a function of k\n" + title
    plt.title(silhouette_title)
    plt.figure()
    plt.plot(np.arange(2, k+2), costs)
    elbow_title = "Elbow The best cost function as a function of k\n" + title
    plt.title(elbow_title)
    plt.show()


def _test_microarray_data(data, is_spectral=True):

    # look at the entire data set:
    k = 5
    if is_spectral:
        title = "Spectral"
        clustering = spectral(data, k, 11)[0]
    else:
        title = "K-Means"
        clustering = run_n_times_kmeans(data, k)[0]

    fig = plt.figure()
    fig.suptitle(title + ": with clustering = "+str(k))
    elem = "3"

    for i in range(k):
        plt.subplot("2" + elem + str(i+1))
        plt.imshow(data[np.where(clustering == i)], extent=[0, 1, 0, 1],
                   cmap="hot", vmin=-3, vmax=3)
        plt.colorbar()
        plt.title("cluster number " + str(i+1)+": "+str(len(np.where(clustering == i)[0])))
    plt.show()


def _test_Tsne(synthetic_data):
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    tsne2 = tsne.TSNE()
    trained_data = tsne2.fit_transform(data)
    plot_with_images(trained_data, data, "Digits example- T-SNE")

    trained_data2 = tsne2.fit_transform(synthetic_data)

    plt.figure()
    plt.title("Synthetic data- TSNE")
    plt.scatter(trained_data2[:,0], trained_data2[:,1])
    plt.show()


def plot_with_images(X, images, title, image_num=50):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)


def _spectral_reduction(data):
    reducted_data = spectral(data, 1, 10, is_dim_reduction=True)
    sorted_indices = np.argsort(reducted_data, axis=0)

    plt.figure()
    plt.title("Spectral reduction on Biological data")
    plt.imshow(data[np.ravel(sorted_indices)], extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()


def visualize_clustering(data, k, sigma):

    res_spectral = spectral(data, k, sigma)
    res_k_mean = run_n_times_kmeans(data, k)
    plt.figure()
    plt.subplot("211")
    plt.scatter(np.ravel(data[:,0]), np.ravel(data[:,1]), c=res_spectral[0])
    plt.title("Spectral with k= "+str(k))
    plt.subplot("212")
    plt.scatter(np.ravel(data[:,0]), np.ravel(data[:,1]), c=res_k_mean[0])
    plt.title("K-Mean with k= "+str(k))
    plt.show()

if __name__ == '__main__':

    # data sets:
    circles = circles_example()
    apml_pic = apml_pic_example()
    synthetic_data = datasets.make_blobs(600, n_features=64, centers=4)[0]
    with open('microarray_data.pickle', 'rb') as f:
        microarray = pickle.load(f)

    # test the spectral algorithm on synthetic data
    visualize_clustering(synthetic_data, 4, 15)
    visualize_clustering(apml_pic, 8, 10)
    visualize_clustering(circles.T, 4, 0.1)

    # measures for K selection on synhetic data with k-means
    _run_examples(synthetic_data, 10, 3, "with synthetic_data")

    _test_microarray_data(microarray)
    _test_microarray_data(microarray, False)
    _test_Tsne(synthetic_data)
    _spectral_reduction(microarray)
