# Preamble
This repository contains the lecture slides and course description for the [Deep Natural Language Processing](http://www.cs.ox.ac.uk/teaching/courses/2016-2017/dl/) course offered in Hilary Term 2017 at the University of Oxford. 

This is an advanced course on natural language processing. Automatically processing natural language inputs and producing language outputs is a key component of Artificial General Intelligence. The ambiguities and noise inherent in human communication render traditional symbolic AI techniques ineffective for representing and analysing language data. Recently statistical techniques based on neural networks have achieved a number of remarkable successes in natural language processing leading to a great deal of commercial and academic interest in the field

This is an applied course focussing on recent advances in analysing and generating speech and text using recurrent neural networks. We introduce the mathematical definitions of the relevant machine learning models and derive their associated optimisation algorithms. The course covers a range of applications of neural networks in NLP including analysing latent dimensions in text, transcribing speech to text, translating between languages, and answering questions. These topics are organised into three high level themes forming a progression from understanding the use of neural networks for sequential language modelling, to understanding their use as conditional language models for transduction tasks, and finally to approaches employing these techniques in combination with other mechanisms for advanced applications. Throughout the course the practical implementation of such models on CPU and GPU hardware is also discussed.

This course was organised by Phil Blunsom and delivered in partnership with the DeepMind Natural Language Research Group. L

# Lecturers
* Phil Blunsom (Oxford University and DeepMind)
* Chris Dyer (Carnegie Mellon University and DeepMind)
* Edward Grefenstette (DeepMind)
* Karl Moritz Hermann (DeepMind)
* Andrew Senior (DeepMind)
* Wang Ling (DeepMind)
* Jeremy Appleyard (NVIDIA)

# Lectures
### 1. Lecture 1a - Introduction [Phil Blunsom]
This lecture introduces the course and motivates why it is interesting to study language processing using Deep Learning techniques.

[[slides]](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%201a%20-%20Introduction.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=ff91caf5-fa7c-42de-8b3d-41f4bc2365b4)

### 2. Lecture 1b - Deep Neural Networks Are Our Friends [Wang Ling]
This lecture revises basic machine learning cocepts that students should know before embarking on this course.

[[slides]](Lecture 1b - Deep Neural Networks Are Our Friends.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=b7d66d78-0deb-46d5-bc14-b1852b9d95e8)

### 3. Lecture 2a- Word Level Semantics [Ed Grefenstette]
Words are the core meaning bearing units in language. Representing and learning the meanings of words is a fundamental task in NLP and in this lecture the concept of a word embedding is introduced as a practical and scalable solution.

[[slides]](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%202a-%20Word%20Level%20Semantics.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=b8531095-9de9-4d4e-bebd-e4272b90ca39)

### 4. Lecture 2b - Overview of the Practicals [Chris Dyer]
This lecture motivates the practical segment of the course.

[[slides]](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%202b%20-%20Overview%20of%20the%20Practicals.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=2ddf7182-43db-44f5-b62a-45e0dfa4f37b)

### 5. Lecture 3 - Language Modelling and RNNs Part 1 [Phil Blunsom]
Language modelling is important task of great practical use in many NLP applications. This lecture introduces language modelling, including traditional n-gram based approaches and more contemporary neural approaches. In particular the popular Recurrent Neural Network (RNN) language model is introduced and its basic training and evaluation algorithms described.

[[slides]](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%203%20-%20Language%20Modelling%20and%20RNNs%20Part%201.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=6bf19af4-d7b3-4ac9-89a1-b719bdd0c2bd)

### 6. Lecture 4 - Language Modelling and RNNs Part 2 [Phil Blunsom]
This lecture continues on from the previous one and considers some of the issues involved in producing an effective implementation of an RNN language model. The vanishing and exploding gradient problem is described and architectural solutions, such as Long Short Term Memory (LSTM), are introducted. 

[[slides]](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%204%20-%20Language%20Modelling%20and%20RNNs%20Part%202.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=fa8df3a8-e7e5-4044-9199-751bcf0a9298)
