import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances

def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target
    #
    plot_with_images(MDS(euclidean_distances(data), 2), data, "Digits example- MDS")
    plot_with_images(LLE(data, 2, 225), data, "Digits example - LLE")
    plot_with_images(DiffusionMap(data, 2, 12, 2, 1000), data,
                     "Digits example - Diffusion map")  # 12 is good

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()

def swiss_roll_example():
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    # LLE Swiss roll-
    Y = LLE(X, 2, 100)
    fig1 = plt.figure()
    ay = fig1.add_subplot(111)
    ay.set_title("Swiss roll example LLE - k 100")

    ay.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.show()

    # DiffusionMap Swiss roll-
    Z = DiffusionMap(X, 2, 10, 2, 30)
    fig2 = plt.figure()
    az = fig2.add_subplot(111)
    az.set_title("Swiss roll example DM - k 30 ")
    az.scatter(Z[:, 0], Z[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.show()

    # MDS Swiss roll-
    mat = euclidean_distances(X,X)
    W = MDS(mat, 2)
    fig3 = plt.figure()
    aw = fig3.add_subplot(111)
    aw.set_title("Swiss roll example MDS ")
    aw.scatter(W[:, 0], W[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()

def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    plot_with_images(MDS(euclidean_distances(X, X), 2), X, "faces_example MDS", 50)
    plot_with_images(LLE(X, 2, 225), X, "faces_example LLE ", 50)
    plot_with_images(DiffusionMap(X, 2, 40, 5, 50), X, "faces_example Diffusion map", 50)  # 40 is good
    num_images, num_pixels = np.shape(X)
    d = int(num_pixels**0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))


    # put this in comment because it runs over my last plot
    # plot some examples of faces:
    # plt.gray()
    # for i in range(4):
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()


def plot_with_images(X, images, title, image_num=25):
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

    return fig


def MDS(X, d, is_eigan_val=False):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param is_eigan_val: if need to return the eigan values or thr matrix
    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix or the eigan values.
    '''

    #  I added the option for this function to return the eigan values (did
    # sort of overloading) so the scree-plot can use this MDS also

    # for the squared distance matrix (because we get only a distance matrix
    X = X**2
    n_by_n = X.shape
    n = n_by_n[0]

    matrix_of_ones = np.ones(n_by_n)
    identity_matrix = np.identity(n)

    H = identity_matrix - (1/n) * matrix_of_ones

    Similarity_Matrix = -0.5 * np.dot(H, np.dot(X, H))

    # take the eigan values and vectors from the similaity matrix
    e_values, e_vectors = LA.eigh(Similarity_Matrix)
    e_values = np.flip(e_values, 0)

    # to get it in descending order
    e_vectors = np.flip(e_vectors, 1)
    n_d_matrix = np.multiply(np.sqrt(e_values), e_vectors)

    if is_eigan_val:
        return  e_values
    return n_d_matrix[:, 0:d]


def _KNN(X, k):
    dis_matrix = euclidean_distances(X,X)
    return [dis_matrix, np.argsort(dis_matrix, 1)[:, :k]]



def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''

    n_by_n = X.shape
    n_dim = n_by_n[0]
    vector_of_ones = np.ones(k)
    identity_matrix = np.identity(n_dim)

    W = np.zeros((n_dim,n_dim))

    knn_indices = _KNN(X, k)[1]
    # create the W matrix
    for i in range(len(X)):
        distance = X[knn_indices[i]] - X[i]
        gram_matrix = distance.dot(distance.transpose())
        inverse_gram = np.linalg.pinv(gram_matrix)
        W[i, knn_indices[i]] = inverse_gram.dot(vector_of_ones)
        W[i] = W[i]/np.sum(W[i])

    M = np.transpose(identity_matrix - W).dot(identity_matrix - W)

    e_values, e_vectors = LA.eigh(M)
    sorted_e_values_indices = e_values.argsort()

    return e_vectors[sorted_e_values_indices][:, 1:d+1]

def DiffusionMap(X, d, sigma, t, k=-1):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    gram matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the gram matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :param k: the amount of neighbors to take into account when calculating the gram matrix.
    :return: Nxd reduced data matrix.
    '''

    N = X.shape[0]

    # if I didn't get a K
    if k == -1:
        k = N

    K = np.zeros((N, N))
    D = np.zeros(N)


    dis_matrix, indices = _KNN(X,k)

    for i in range(N):
        K[i, indices[i]] = np.exp(-(dis_matrix[i, indices[i]])**2/sigma)
        D[i] = np.sum(K[i])

    # normalize D
    diag_D = np.diag(D)
    A = np.linalg.inv(diag_D).dot(K)

    e_values, e_vectors = LA.eigh(A)
    # sort the eigan values and also power it by t
    sorted_e_values = np.flip(e_values, 0)[1:d+1]**t
    sorted_e_vec = np.flip(e_vectors, 1)[:,1:d+1]

    return np.multiply(sorted_e_values, sorted_e_vec)


def _scree_plot(epsilon):
    # chose a random dimension for the gaussian matrix
    gaussian = np.random.normal(size=(100,2))
    zero_matrix = np.zeros((100,98))
    expanded_gaussian = np.concatenate((gaussian, zero_matrix), 1)

    q, r = LA.qr(expanded_gaussian)

    # Gaussian noise matrix
    noise_gaussian = np.random.normal(scale= epsilon, size=(100,100))
    noised = np.dot(expanded_gaussian, q) + noise_gaussian

    res = MDS(euclidean_distances(noised), 2, True)
    fig = plt.figure()
    aw = fig.add_subplot(111)
    aw.set_title("scree plot with noise: %.1f" %epsilon)
    plt.plot(res, 'o')
    plt.show()

def _lossyCompression(res_dim):
    gaussian = np.random.normal(size=(64,64))
    distance_gaussian = euclidean_distances(gaussian)
    vec_distance_gaussian = np.reshape(distance_gaussian, (1,64*64))[0]

    res = MDS(distance_gaussian, res_dim)
    res_distance = euclidean_distances(res)
    vec_res = np.reshape(res_distance, (1,res_distance.shape[0]*res_distance.shape[0]))[0]
    fig = plt.figure()
    aw = fig.add_subplot(111)
    aw.set_title("Lossy compression with dimension: %d" %res_dim)
    plt.plot(vec_distance_gaussian, vec_res, 'o')
    plt.show()

if __name__ == '__main__':
    digits_example()
    swiss_roll_example()
    faces_example("faces.pickle")
    _scree_plot(0)
    _scree_plot(0.2)
    _scree_plot(0.6)
    _scree_plot(1)
    _lossyCompression(64)
    _lossyCompression(50)
    _lossyCompression(40)
    _lossyCompression(30)
    _lossyCompression(10)
    _lossyCompression(5)
