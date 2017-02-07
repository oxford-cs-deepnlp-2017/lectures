# Preamble
This repository contains the lecture slides and course description for the [Deep Natural Language Processing](http://www.cs.ox.ac.uk/teaching/courses/2016-2017/dl/) course offered in Hilary Term 2017 at the University of Oxford. 

This is an advanced course on natural language processing. Automatically processing natural language inputs and producing language outputs is a key component of Artificial General Intelligence. The ambiguities and noise inherent in human communication render traditional symbolic AI techniques ineffective for representing and analysing language data. Recently statistical techniques based on neural networks have achieved a number of remarkable successes in natural language processing leading to a great deal of commercial and academic interest in the field

This is an applied course focussing on recent advances in analysing and generating speech and text using recurrent neural networks. We introduce the mathematical definitions of the relevant machine learning models and derive their associated optimisation algorithms. The course covers a range of applications of neural networks in NLP including analysing latent dimensions in text, transcribing speech to text, translating between languages, and answering questions. These topics are organised into three high level themes forming a progression from understanding the use of neural networks for sequential language modelling, to understanding their use as conditional language models for transduction tasks, and finally to approaches employing these techniques in combination with other mechanisms for advanced applications. Throughout the course the practical implementation of such models on CPU and GPU hardware is also discussed.

This course is organised by Phil Blunsom and delivered in partnership with the DeepMind Natural Language Research Group.

# Lecturers
* Phil Blunsom (Oxford University and DeepMind)
* Chris Dyer (Carnegie Mellon University and DeepMind)
* Edward Grefenstette (DeepMind)
* Karl Moritz Hermann (DeepMind)
* Andrew Senior (DeepMind)
* Wang Ling (DeepMind)
* Jeremy Appleyard (NVIDIA)

# Lectures
## 1. Lecture 1a - Introduction [Phil Blunsom]
This lecture introduces the course and motivates why it is interesting to study language processing using Deep Learning techniques.

[[slides]](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%201a%20-%20Introduction.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=ff91caf5-fa7c-42de-8b3d-41f4bc2365b4)

## 2. Lecture 1b - Deep Neural Networks Are Our Friends [Wang Ling]
This lecture revises basic machine learning concepts that students should know before embarking on this course.

[[slides]](Lecture 1b - Deep Neural Networks Are Our Friends.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=b7d66d78-0deb-46d5-bc14-b1852b9d95e8)

## 3. Lecture 2a- Word Level Semantics [Ed Grefenstette]
Words are the core meaning bearing units in language. Representing and learning the meanings of words is a fundamental task in NLP and in this lecture the concept of a word embedding is introduced as a practical and scalable solution.

[[slides]](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%202a-%20Word%20Level%20Semantics.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=b8531095-9de9-4d4e-bebd-e4272b90ca39)

## 4. Lecture 2b - Overview of the Practicals [Chris Dyer]
This lecture motivates the practical segment of the course.

[[slides]](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%202b%20-%20Overview%20of%20the%20Practicals.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=2ddf7182-43db-44f5-b62a-45e0dfa4f37b)

## 5. Lecture 3 - Language Modelling and RNNs Part 1 [Phil Blunsom]
Language modelling is important task of great practical use in many NLP applications. This lecture introduces language modelling, including traditional n-gram based approaches and more contemporary neural approaches. In particular the popular Recurrent Neural Network (RNN) language model is introduced and its basic training and evaluation algorithms described.

[[slides]](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%203%20-%20Language%20Modelling%20and%20RNNs%20Part%201.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=6bf19af4-d7b3-4ac9-89a1-b719bdd0c2bd)

### Reading

#### Textbook
 * [Deep Learning, Chapter 10](http://www.deeplearningbook.org/contents/rnn.html).
 
#### Blogs 
 * [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), Andrej Karpathy.
 * [The unreasonable effectiveness of Character-level Language Models](http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139), Yoav Goldberg.
 * [Explaining and illustrating orthogonal initialization for recurrent neural networks](http://smerity.com/articles/2016/orthogonal_init.html), Stephen Merity.

## 6. Lecture 4 - Language Modelling and RNNs Part 2 [Phil Blunsom]
This lecture continues on from the previous one and considers some of the issues involved in producing an effective implementation of an RNN language model. The vanishing and exploding gradient problem is described and architectural solutions, such as Long Short Term Memory (LSTM), are introducted. 

[[slides]](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%204%20-%20Language%20Modelling%20and%20RNNs%20Part%202.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa8df3a8-e7e5-4044-9199-751bcf0a9298)

### Reading

#### Textbook
 * [Deep Learning, Chapter 10](http://www.deeplearningbook.org/contents/rnn.html).
 
#### Vanishing gradients, LSTMs etc.
* [On the difficulty of training recurrent neural networks. Pascanu et al., ICML 2013.](http://jmlr.csail.mit.edu/proceedings/papers/v28/pascanu13.pdf)
* [Long Short-Term Memory. Hochreiter and Schmidhuber, Neural Computation 1997.](http://dl.acm.org/citation.cfm?id=1246450)
* [Learning Phrase Representations using RNN EncoderDecoder for Statistical Machine Translation. Cho et al, EMNLP 2014.](https://arxiv.org/abs/1406.1078)
* Blog: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), Christopher Olah.

#### Dealing with large vocabularies
* [A scalable hierarchical distributed language model. Mnih and Hinton, NIPS 2009.](https://papers.nips.cc/paper/3583-a-scalable-hierarchical-distributed-language-model.pdf)
* [A fast and simple algorithm for training neural probabilistic language models. Mnih and Teh, ICML 2012.](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf)
* [On Using Very Large Target Vocabulary for Neural Machine Translation. Jean et al., ACL 2015.](http://www.aclweb.org/anthology/P15-1001)
* [Exploring the Limits of Language Modeling. Jozefowicz et al., arXiv 2016.](https://arxiv.org/abs/1602.02410)
* [Efficient softmax approximation for GPUs. Grave et al., arXiv 2016.](https://arxiv.org/abs/1609.04309)
* [Notes on Noise Contrastive Estimation and Negative Sampling. Dyer, arXiv 2014.](https://arxiv.org/abs/1410.8251)
* [Pragmatic Neural Language Modelling in Machine Translation. Baltescu and Blunsom, NAACL 2015](http://www.aclweb.org/anthology/N15-1083)

#### Regularisation and dropout
* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks. Gal and Ghahramani, NIPS 2016.](https://arxiv.org/abs/1512.05287)
* Blog: [Uncertainty in Deep Learning](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html), Yarin Gal.

#### Other stuff
* [Recurrent Highway Networks. Zilly et al., arXiv 2016.](https://arxiv.org/abs/1607.03474)
* [Capacity and Trainability in Recurrent Neural Networks. Collins et al., arXiv 2016.](https://arxiv.org/abs/1611.09913)
* [Optimizing Performance of Recurrent Neural Networks on GPUs. Appleyard et al., arXiv 2016.](https://arxiv.org/abs/1604.01946)

# Piazza
We will be using Piazza to facilitate class discussion during the course. Rather than emailing questions directly, I encourage you to post your questions on Piazza to be answered by your fellow students, instructors, and lecturers. However do please do note that all the lecturers for this course are volunteering their time and may not always be available to give a reponse.

Find our class page at: https://piazza.com/ox.ac.uk/winter2017/dnlpht2017/home

# Assessment
The primary assessment for this course will be a take-home assignment issued at the end of the term. This assignment will ask questions drawing on the concepts and models discussed in the course, as well as from selected research publications. The nature of the questions will include analysing mathematical descriptions of models and proposing extensions, improvements, or evaluations to such models. The assignment may also ask students to read specific research publications and discuss their proposed algorithms in the context of the course. In answering questions students will be expected to both present coherent written arguments and use appropriate mathematical formulae, and possibly pseudo-code, to illustrate answers.

The practical component of the course will be assessed in the usual way.
