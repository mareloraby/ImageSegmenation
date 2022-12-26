import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape # N: Number of examples - D: number of features

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False) # chooses k random indices less than N as clusters and returns an array of the random indices
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    ### YOUR CODE HERE
    for _ in range(num_iters):
        clusters = create_cluster(features, centers, k) # create cluster
        previous_centers = centers
        centers = calculate_new_centroids(features,clusters,k)
        diff = centers - previous_centers
        if not diff.any():
            break # stop when centroids are optimized
        # pass
    assignments = predict_cluster(clusters, N)
    ### END YOUR CODE

    return assignments


def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part (cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

        ### YOUR CODE HERE
    for _ in range(num_iters):
        clusters = create_cluster_fast(features, centers, k) # create cluster
        previous_centers = centers
        centers = calculate_new_centroids(features,clusters,k)
        diff = centers - previous_centers
        if not diff.any():
            break # stop when centroids are optimized
        # pass
    assignments = predict_cluster(clusters, N)
    ### END YOUR CODE

    return assignments


# helper methods for kmeans:

# assign points to clusters
def create_cluster(features, centers,k):
    clusters = [[] for _ in range(k)]
    for point_idx, point in enumerate(features):
        closest_centroid = np.argmin(
            np.sqrt(np.sum((point-centers)**2, axis=1))
        ) # closest centroid using euclidean distance
        clusters[closest_centroid].append(point_idx)
    return clusters 

# assign points to clusters fast
def create_cluster_fast(features, centers,k):
    clusters = [[] for _ in range(k)]
    dists = cdist(features, centers, metric='euclidean') 
    result = np.argmin(dists,1)
    for point_idx, assigned_cluster in enumerate(result):
        clusters[assigned_cluster].append(point_idx)
    return clusters 

# new centroids
def calculate_new_centroids(features,clusters,k):
    centers = np.zeros((k, features.shape[1]))
    for idx, cluster in enumerate(clusters):
        new_centroid = np.mean(features[cluster], axis=0) # find new centroids
        centers[idx] = new_centroid

    return centers 

 # prediction
def predict_cluster(clusters, N):
    y_pred = np.zeros(N, dtype=np.uint32) 
    for cluster_idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            y_pred[sample_idx] = cluster_idx
            
    return y_pred           

# End of helper methods of Kmeans

def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Hints
    - You may find pdist (imported from scipy.spatial.distance) useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N
    

    while n_clusters > k:
        ### YOUR CODE HERE
        pass
        ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    features = img.reshape(H*W, C)
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    
    imageIndices = np.dstack(np.mgrid[0:H,0:W])
    # print(color[0][0])
    # print(color[0][0][0])
    # print(img[0].size)
    # print(imageIndices[0])

    n = 0
    for i in range (0,H):
      for j in range (0,W):
        features[n][0] = color[i][j][0]
        features[n][1] = color[i][j][1]
        features[n][2] = color[i][j][2]
        features[n][3] = imageIndices[i][j][0]
        features[n][4] = imageIndices[i][j][1]
        n +=1
    
    features -= np.mean(features, axis=0)   #set mean to zero
    features /= np.std(features, axis=0)
      
    ### END YOUR CODE
    return features

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    isEqual = mask_gt == mask
    accuracy = np.mean(isEqual)
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
