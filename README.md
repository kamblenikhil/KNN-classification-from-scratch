# K-Nearest Neighbors Classification

## What is K-Nearest Neighbors (KNN)
K-Nearest Neighbors evaluates the labels of a certain number of data points surrounding a target data point in order to forecast which class the data point belongs to. <br>

When a KNN algorithm is run, it passes through three basic stages:

1. Set K to the number of neighbors you want.
2. The distance between a provided/test example and the dataset examples is calculated.
3. The calculated distances are sorted.
4. Obtaining the top K entries' labels.
5. A prediction about the test example is returned.

![image](https://media.github.iu.edu/user/18330/files/64f1de5a-ad92-4003-8599-43c3bcaf66a5)

Here, we test the given data by calculating Manhattan and Euclidean distance as per the metrics are passed to the knn function using the fitted classifier model. 

```python3
    # find distance: if l1 ->manhattan, else -> euclidean
    if self.metric == "l1":
      distance = manhattan_distance(self._X[j], X[i])
    elif self.metric == "l2":
      distance = euclidean_distance(self._X[j], X[i])
    
    d.append([distance, self._y[j]])
```

<b>Euclidean Distance</b>
```python3
    eu_distance = np.sqrt(((x1-x2)**2).sum())
```

<b>Manhattan Distance</b>
```python3
    man_distance = np.abs(x1 - x2).sum()
```

Once we get all the distances, we sort the array and consider n_neighbors (3, 5, 7, 11) to then weigh the class considering the weights as uniform and distance. After weighing the class, we then append the maximum value from the given dictionary to our final prediction list.

```python3

# sorting the distance array
d.sort()

# consider only n neighbors
for x in range(0,self.n_neighbors):
    a.append(d[x])

# weighing the class
for dist, clust in a:
    # considering weights as uniform and distance 
    if self.weights == "uniform":
        if clust not in temp:
            temp[clust] = 1
        else:
            temp[clust] = temp[clust] + 1
    # the below code was suggested by stephen
    elif self.weights == "distance":
        if clust not in temp:
            temp[clust] = float(1/dist)
        else:
            temp[clust] = temp[clust] + float(1/dist)

# appending the max from the given temp dictionary
prediction.append(max(temp,key=temp.get))
```
