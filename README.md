## Roadmap

All task assignments are not final and subject to change. Inter-group collaboration is always welcome.

### Basis of everything (Jiahang, Eddie)

- [ ] Finding a pretrained model to classify political ideology.
- [ ] Use 4o-mini to label some data and finetune the existing model.
- [ ] Also create a few baseline embedding/tf-idf + tree-based/logistic regression models for comparison.

### Comparison of article and outlet biases (Jiahang, Eddie)

- [ ] Choose a few news outlets, both mainstream and alternative, plot the distribution of article ideology, and compare the said distribution with the outlet ideology ratings.
- [ ] Identify certain political domains of interest, such as abortion, LGBTQ, gun rights, Sino-US relations, Russian, plot the polarization of articles in these domains.

### Quotation analysis (Peter, Lexie)

- [ ] Extract quotations from articles.
- [ ] Compare the distribution of quoted sources with the article ideology.
- [ ] Build models to predict what kind of sources are quoted by left-leaning, right-leaning, neutral outlets, or are commonly quoted by all outlets.

### Repost network analysis (Max, Ricky)

- [ ] Create repost networks with edge weights representing article similarity.
- [ ] Use the network to identify the most influential articles and outlets.
- [ ] Use semantic embeddings instead of article similarity to construct the repost network, consider the influence of mainstream media on alternative media and vice versa.

### News coverage network (Max, Ricky)

- [ ] Create a bipartite network of articles and outlets, with edges representing the publication of articles by outlets, and edge weights representing the ideology of the article.
- [ ] Use GNN or autoencoder to predict the connection between articles and outlets, and the ideology of the article as edge weights.
- [ ] Potentially use the embedding of the existing articles as edge attributes, predict the edge attributes of new articles, and decode the embedding to predict the article text.
