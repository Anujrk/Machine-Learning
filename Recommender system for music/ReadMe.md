 I used the Spotify Dataset, which is publicly available on Kaggle and contains metadata and audio features for over 170,000 different songs.
 LINK : https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks
 The "data.csv" file contains more than 175.000 songs collected from Spotify Web API, and also you can find data grouped by artist, year, or genre in the data section.
 
In the code, I used the Expectation Maximisation using Gaussian mixture model which is a representation of a Gaussian mixture model probability distribution to divide the over 2,900 genres in this dataset into twenty clusters based on the numerical audio features of each genre.
Unlike Kmeans algorithm, EM is not sensitive to the choice of distance metric and no need to specify the number of clusters. You have the option of choosing the best-looking clusters.

Dimensionality reduction is carried out to reduce the genre data frame  by using t-distributed stochastic neighbor embedding and  also by using Principal component analysis (PCA) which gives us a result that PCA is more efficient and faster then t-SNE.

The algorithm to find next recommendation is based on cosine distance, which is defined for two vectors x and y as :
Distance(x, y) = 1- (x . y) / ||x|| * ||y||
               = 1- Cosθ
The cosine distance is commonly used in recommender systems and can work well even when the vectors being used have different magnitudes. If the vectors for two songs are parallel, the angle between them will be zero, meaning the cosine distance between them will also be zero because the cosine of zero is 1.
The Logic behind the algorithm is filtering songs with same Genre cluster and If song count is one in that cluster,we return that song name else if count of song is greter than 1, we calculate the distance and return songs with smallest distance.

