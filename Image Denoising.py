import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import pickle
from skimage.util import view_as_windows as viewW
import time

def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(patches[:,i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                        window[0] * window[1]).T[:, ::stepsize]


def greyscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = greyscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
        noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original - denoised_images[i]) ** 2))

    plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.show()


class GMM_Model:
    """
    A class that represents a Gaussian Mixture Model, with all the parameters
    needed to specify the model.

    mixture - a length k vector with the multinomial parameters for the gaussians.
    means - a k-by-D matrix with the k different mean vectors of the gaussians.
    cov - a k-by-D-by-D tensor with the k different covariance matrices.
    """
    def __init__(self, mixture, means, cov):
        self.mixture = mixture
        self.means = means
        self.cov = cov


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D siznormalize_log_likelihoodsed vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    gmm - a GMM_Model object.
    """
    def __init__(self, means, cov, gmm, log_likelihood_history):
        self.means = means
        self.cov = cov
        self.gmm = gmm
        self.log_likelihood_history = log_likelihood_history


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    gmm - a GMM_Model object.
    """
    def __init__(self, cov, mixture, gmm, log_likelihood_history):
        self.cov = cov
        self.mixture = mixture
        self.gmm = gmm
        self.log_likelihood_history = log_likelihood_history


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    gmms - A list of K GMM_Models, one for each source.
    """
    def __init__(self, P, vars, mixture, gmms, log_likelihood_history):
        self.P = P
        self.vars = vars
        self.mixture = mixture
        self.gmms = gmms
        self.log_likelihood_history = log_likelihood_history


def MVN_log_likelihood(X, gmm_model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """
    return _log_likelihood(X, gmm_model)

def _log_likelihood(X, gmm_model):
    k = gmm_model.cov.shape[0]
    N = X.shape[1]
    C = np.zeros((N, k))

    for y in range(k):
        C[:, y] = np.log(gmm_model.mixture[y]) + \
                  multivariate_normal(gmm_model.means[y], gmm_model.cov[y],
                                      allow_singular=True).logpdf(X.T)

    return np.sum(np.multiply(calculate_C(X, gmm_model.cov, k, gmm_model.means, gmm_model.mixture), C))

def GSM_log_likelihood(X, gmm_model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """
    return _log_likelihood(X, gmm_model)



def ICA_log_likelihood(X, gmm_models):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """
    N = X.shape[1]
    D = X.shape[0]
    # calculate A
    cov = np.cov(X)
    A = np.linalg.eig(cov)[1]
    S = np.dot(A.T, X)
    res = 0
    for i in range(D):
        res += _log_likelihood(S[i].reshape((1,N)), gmm_models[i])
    return res


def EM(X, means, cov, mixture, k, is_GSM, learn_mixture=True, learn_means=True,
              learn_covariances=True, iterations=10,  r=None):
    """
    A general function for learning a GMM_Model using the EM algorithm.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: number of components in the mixture.
    :param initial_model: an initial GMM_Model object to initialize EM with.
    :param learn_mixture: a boolean for whether or not to learn the mixtures.
    :param learn_means: a boolean for whether or not to learn the means.
    :param learn_covariances: a boolean for whether or not to learn the covariances.
    :param iterations: Number of EM iterations (default is 10).
    :return: (GMM_Model, log_likelihood_history)
            GMM_Model - The learned GMM Model.
            log_likelihood_history - The log-likelihood history for debugging.
    """
    N = X.shape[1]
    D = X.shape[0]
    log_likelihood_history = []
    cov_kdd = np.zeros((k, D, D))
    r_squared = np.zeros((k,1))
    if is_GSM:
        for y in range(k):
            cov_kdd[y] = r[y] * cov
    else:
        cov_kdd = cov
    for i in range(iterations):

        C = calculate_C(X, cov_kdd, k, means, mixture)
        sum_C_col = np.sum(C, 0)

        if learn_mixture:
            mixture = sum_C_col / N
        if learn_means:
            means = np.divide(np.dot(X, C), sum_C_col).T
        if learn_covariances:
            for y in range(k):
                mat = np.subtract(X.T, means[y]).T

                cov_kdd[y] = np.dot(np.multiply(C[:, y], mat),
                                np.transpose(mat)) / np.sum(C[:, y])

        if is_GSM:
            mat = np.diag(np.dot(np.dot(X.T, np.linalg.pinv(cov)), X))
            for y in range(k):
                r_squared[y] = np.dot(C.T[y], mat)
                norm = D * np.sum(C.T[y])
                r_squared[y] = r_squared[y]/norm
                cov_kdd[y] = r_squared[y] * cov

        log_likelihood_history.append(_log_likelihood(X, GMM_Model(
            mixture, means, cov_kdd)))

    return GMM_Model(mixture.copy(), means.copy(), cov_kdd.copy()), log_likelihood_history


def calculate_C(X, cov_kdd, k, means, mixture, noise_std=0):
    N = X.shape[1]
    C = np.zeros((N, k))
    for y in range(k):
        C[:, y] = np.log(mixture[y]) + \
                  multivariate_normal(means[y], cov_kdd[y] +
                                      (noise_std ** 2) * np.identity(X.shape[0]),
                                      allow_singular=True).logpdf(X.T)
    C = np.exp(normalize_log_likelihoods(C.T)).T
    return C


def learn_MVN(X):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :return: A trained MVN_Model object.
    """
    start = time.time()

    # N = number of samples, d = dimension
    N = X.shape[1]
    d = X.shape[0]
    cov_kdd = np.zeros((1, d, d))
    means_kd = np.zeros((1, d))
    means = np.sum(X, 1)/N
    means_kd[0] = means
    cov = np.cov(X)
    cov_kdd[0] = cov
    gmm = GMM_Model(np.array([1]), means_kd, cov_kdd)
    end = time.time()
    print("in learn_MVN: ", (end - start))

    return MVN_Model(means, cov, gmm, MVN_log_likelihood(X, gmm))


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """
    start = time.time()
    mixture = np.ones(k)/k

    r = np.arange(1, k+1)
    D = X.shape[0]
    # means is a matrix of zeros
    means = np.zeros((k,D))
    gmm, log_likelihood_history = EM(X, means, np.cov(X), mixture, k, True, learn_means=False,
                                     learn_covariances=False, r=r)
    end = time.time()
    print("in learn_GSM: ",(end - start))

    return GSM_Model(gmm.cov, gmm.mixture, gmm, log_likelihood_history)

def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """
    start = time.time()

    D = X.shape[0]
    N = X.shape[1]
    mixture = np.ones(k)/k
    # calculate A
    cov = np.cov(X)
    A = np.linalg.eig(cov)[1]
    k_cov = np.zeros((k, 1, 1))
    S = np.dot(A.T, X)
    log_likelihood_history = None

    var_list = np.zeros((D, k))
    mixs = np.zeros((D, k))
    gmms = [None] * D
    for i in range(D):
        cov_s = np.cov(S[i])
        for j in range(1, k+1):
            k_cov[j-1, 0, 0] = j * cov_s

        gmms[i], log_likelihood_history = EM(S[i].reshape((1, N)),
                                         np.zeros((k,1)), k_cov, mixture, k,
                                         False, learn_means=False)
        var_list[i] = gmms[i].cov.reshape((k,))
        mixs[i] = gmms[i].mixture.copy()
    end = time.time()
    print("in learn_ICA: ",(end - start))
    return ICA_Model(A, var_list, mixs, gmms, log_likelihood_history)

def _Weiner(Y, cov, means, noise_std):

    D = Y.shape[0]
    left_side = np.linalg.pinv(np.linalg.pinv(cov) + 1/(noise_std**2) * np.identity(D))
    right_side = np.linalg.pinv(cov).dot(means) + np.transpose(1/(noise_std**2) * Y)
    return left_side.dot(right_side.T)


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    start = time.time()
    denoised = _Weiner(Y, mvn_model.cov, mvn_model.means, noise_std)
    end = time.time()
    print("in MVN_Denoise: ",(end - start))
    return denoised


def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """
    start = time.time()

    X_star = np.zeros(Y.shape)
    D = Y.shape[0]
    k = gsm_model.mixture.shape[0]
    means = np.zeros((k,D))
    C = calculate_C(Y, gsm_model.cov, k, means, gsm_model.mixture, noise_std)
    for i in range(k):
        X_star += np.multiply(_Weiner(Y, gsm_model.cov[i], means[i], noise_std), C[:,i])
    end = time.time()
    print("in GSM_Denoise: ",(end - start))
    return X_star

def _ICA_Weiner(Y, var, noise_std):
    left_side = 1/(1/var + 1/(noise_std ** 2))
    right_side = 1/(noise_std**2) * Y

    return left_side * right_side


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    start = time.time()
    N = Y.shape[1]
    k = ica_model.mixture.shape[1]
    S_noisy = np.transpose(ica_model.P).dot(Y)
    means = np.zeros((k,1))
    S_denoised = np.zeros(Y.shape)
    D = Y.shape[0]
    for i in range (D):
        C = calculate_C(S_noisy[i].reshape((1, N)), ica_model.vars[i], k, means, ica_model.mixture[i], noise_std)
        for j in range (k):
            S_denoised[i] += np.multiply(_ICA_Weiner(S_noisy[i], ica_model.vars[i, j], noise_std), C[:,j])

    X_denoised = ica_model.P.dot(S_denoised)
    end = time.time()
    print("in ICA_Denoise: ",(end - start))
    return X_denoised


def create_patches():
    patch_size = (8, 8)
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)
    return patches

def create_test_patches():
    patch_size = (8, 8)

    with open('test_images.pickle', 'rb') as f:
        test_pictures = pickle.load(f)
    test_patches = sample_patches(test_pictures, psize=patch_size, n=20000)
    return test_patches, test_pictures

def plot_log_likelihood_history(patches):
    model = learn_GSM(patches, 5)
    fig = plt.figure()
    aw = fig.add_subplot(111)
    aw.set_title("GSM log_likelihood_history:")
    plt.plot(np.arange(10), model.log_likelihood_history)
    plt.show()

if __name__ == '__main__':
    patches = create_patches()
    plot_log_likelihood_history(patches)
    test_patches, test_pictures = create_test_patches()

    standarized = greyscale_and_standardize(test_pictures)

    model = learn_MVN(patches)
    print("MVN_log_likelihood: ", MVN_log_likelihood(test_patches, model.gmm))
    test_denoising(standarized[10], model, MVN_Denoise)

    model = learn_GSM(patches, 5)
    print("GSM_log_likelihood: ", GSM_log_likelihood(test_patches, model.gmm))
    test_denoising(standarized[10], model, GSM_Denoise)

    model = learn_ICA(patches, 5)
    print("ICA_log_likelihood", ICA_log_likelihood(test_patches, model.gmms))
    test_denoising(standarized[10], model, ICA_Denoise)