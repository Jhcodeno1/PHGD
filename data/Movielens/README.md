## MovieLens
(Containing rating and timestamp information)

(Note: We utilize the Pearson's coefficient to measure the similiarities in the KNN algorithm)

(Source : https://grouplens.org/datasets/movielens/)

| Entity         |#Entity        |
| :-------------:|:-------------:|
| User           | 943           |
| Age            | 8             |
| Occupation     | 21            |
| Movie          | 1,682         |
| Genre          | 18            |

### Relation Statistics
| Relation            |#Relation      |
| :-------------:     |:-------------:|
| User - Movie        | 100,000       |
| User - User (KNN)   | 47,150        |
| User - Age          | 943           |
| User - Occupation   | 943           |
| Movie - Movie (KNN) | 82,798        |
| Movie - Genre       | 2,861         |

### User-item sparsity 
0.9369533063577546