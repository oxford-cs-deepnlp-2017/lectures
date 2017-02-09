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

# TAs
* Yannis Assael
* Yishu Miao
* Brendan Shillingford
* Jan Buys

# Timetable
## Practicals
 * Group 1 - Monday, 9:00-11:00 (Weeks 2-8), 60.05 Thom Building
 * Group 2 - Friday, 16:00-18:00 (Weeks 2-8), Room 379

## Lectures
Public Lectures are held in Lecture Theatre 1 of the Maths Institute, on Tuesdays and Thursdays, 16:00-18:00 (Hilary Term Weeks 1,3-8).

# Lecture Materials
## 1. Lecture 1a - Introduction [Phil Blunsom]
This lecture introduces the course and motivates why it is interesting to study language processing using Deep Learning techniques.

[[slides]](Lecture 1a - Introduction.pdf)
[[video]](http://media.podcasts.ox.ac.uk/comlab/deep_learning_NLP/2017-01_deep_NLP_1a_intro.mp4)

## 2. Lecture 1b - Deep Neural Networks Are Our Friends [Wang Ling]
This lecture revises basic machine learning concepts that students should know before embarking on this course.

[[slides]](Lecture 1b - Deep Neural Networks Are Our Friends.pdf)
[[video]](http://media.podcasts.ox.ac.uk/comlab/deep_learning_NLP/2017-01_deep_NLP_1b_friends.mp4)

## 3. Lecture 2a- Word Level Semantics [Ed Grefenstette]
Words are the core meaning bearing units in language. Representing and learning the meanings of words is a fundamental task in NLP and in this lecture the concept of a word embedding is introduced as a practical and scalable solution.

[[slides]](Lecture 2a- Word Level Semantics.pdf)
[[video]](http://media.podcasts.ox.ac.uk/comlab/deep_learning_NLP/2017-01_deep_NLP_2a_lexical_semantics.mp4)

### Reading

#### Embeddings Basics
* [Firth, John R. "A synopsis of linguistic theory, 1930-1955." (1957): 1-32.](http://annabellelukin.edublogs.org/files/2013/08/Firth-JR-1962-A-Synopsis-of-Linguistic-Theory-wfihi5.pdf)
* [Curran, James Richard. "From distributional to semantic similarity." (2004).](https://www.era.lib.ed.ac.uk/bitstream/handle/1842/563/IP030023.pdf?sequence=2&isAllowed=y)
* [Collobert, Ronan, et al. "Natural language processing (almost) from scratch." Journal of Machine Learning Research 12. Aug (2011): 2493-2537.](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)
* [Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

#### Datasets and Visualisation
* [Finkelstein, Lev, et al. "Placing search in context: The concept revisited." Proceedings of the 10th international conference on World Wide Web. ACM, 2001.](http://www.iicm.tugraz.at/thesis/cguetl_diss/literatur/Kapitel07/References/Finkelstein_et_al._2002/p116-finkelstein.pdf)
* [Hill, Felix, Roi Reichart, and Anna Korhonen. "Simlex-999: Evaluating semantic models with (genuine) similarity estimation." Computational Linguistics (2016).](http://www.aclweb.org/website/old_anthology/J/J15/J15-4004.pdf)
* [Maaten, Laurens van der, and Geoffrey Hinton. "Visualizing data using t-SNE." Journal of Machine Learning Research 9.Nov (2008): 2579-2605.](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

#### Blog posts
* [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/), Christopher Olah.
* [Visualizing Top Tweeps with t-SNE, in Javascript](http://karpathy.github.io/2014/07/02/visualizing-top-tweeps-with-t-sne-in-Javascript/), Andrej Karpathy.

#### Further Reading
* [Hermann, Karl Moritz, and Phil Blunsom. "Multilingual models for compositional distributed semantics." arXiv preprint arXiv:1404.4641 (2014).](https://arxiv.org/pdf/1404.4641.pdf)
* [Levy, Omer, and Yoav Goldberg. "Neural word embedding as implicit matrix factorization." Advances in neural information processing systems. 2014.](http://u.cs.biu.ac.il/~nlp/wp-content/uploads/Neural-Word-Embeddings-as-Implicit-Matrix-Factorization-NIPS-2014.pdf)
* [Levy, Omer, Yoav Goldberg, and Ido Dagan. "Improving distributional similarity with lessons learned from word embeddings." Transactions of the Association for Computational Linguistics 3 (2015): 211-225.](https://www.transacl.org/ojs/index.php/tacl/article/view/570/124)
* [Ling, Wang, et al. "Two/Too Simple Adaptations of Word2Vec for Syntax Problems." HLT-NAACL. 2015.](https://www.aclweb.org/anthology/N/N15/N15-1142.pdf)

## 4. Lecture 2b - Overview of the Practicals [Chris Dyer]
This lecture motivates the practical segment of the course.

[[slides]](Lecture 2b - Overview of the Practicals.pdf)
[[video]](http://media.podcasts.ox.ac.uk/comlab/deep_learning_NLP/2017-01_deep_NLP_2b_practicals.mp4)

## 5. Lecture 3 - Language Modelling and RNNs Part 1 [Phil Blunsom]
Language modelling is important task of great practical use in many NLP applications. This lecture introduces language modelling, including traditional n-gram based approaches and more contemporary neural approaches. In particular the popular Recurrent Neural Network (RNN) language model is introduced and its basic training and evaluation algorithms described.

[[slides]](Lecture 3 - Language Modelling and RNNs Part 1.pdf)
[[video]](http://media.podcasts.ox.ac.uk/comlab/deep_learning_NLP/2017-01_deep_NLP_3_modelling_1.mp4)

### Reading

#### Textbook
 * [Deep Learning, Chapter 10](http://www.deeplearningbook.org/contents/rnn.html).
 
#### Blogs 
 * [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), Andrej Karpathy.
 * [The unreasonable effectiveness of Character-level Language Models](http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139), Yoav Goldberg.
 * [Explaining and illustrating orthogonal initialization for recurrent neural networks](http://smerity.com/articles/2016/orthogonal_init.html), Stephen Merity.

## 6. Lecture 4 - Language Modelling and RNNs Part 2 [Phil Blunsom]
This lecture continues on from the previous one and considers some of the issues involved in producing an effective implementation of an RNN language model. The vanishing and exploding gradient problem is described and architectural solutions, such as Long Short Term Memory (LSTM), are introduced. 

[[slides]](Lecture 4 - Language Modelling and RNNs Part 2.pdf)
[[video]](http://media.podcasts.ox.ac.uk/comlab/deep_learning_NLP/2017-01_deep_NLP_4_modelling_2.mp4)

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

## 7. Lecture 5 - Text Classification [Karl Moritz Hermann]
This lecture discusses text classification, beginning with basic classifiers, such as Naive Bayes, and progressing through to RNNs and Convolution Networks.

[[slides]](Lecture 5 - Text Classification.pdf) 
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=dbc0d5cb-4d24-41be-a988-85a08eb1b2f7)

### Reading
 * [Recurrent Convolutional Neural Networks for Text Classification. Lai et al. AAAI 2015.](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552) 
 * [A Convolutional Neural Network for Modelling Sentences, Kalchbrenner et al. ACL 2014.](http://www.aclweb.org/anthology/P14-1062)
 * [Semantic compositionality through recursive matrix-vector, Socher et al. EMNLP 2012.](http://nlp.stanford.edu/pubs/SocherHuvalManningNg_EMNLP2012.pdf)
 * Blog: [Understanding Convolution Neural Networks For NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/), Denny Britz.
 * Thesis: [Distributional Representations for Compositional Semantics, Hermann (2014).](https://arxiv.org/abs/1411.3146)

## 8. Lecture 6 - Deep NLP on Nvidia GPUs [Jeremy Appleyard]
This lecture introduces Graphical Processing Units (GPUs) as an alternative to CPUs for executing Deep Learning algorithms. The strengths and weaknesses of GPUs are discussed as well as the importance of understanding how memory bandwidth and computation impact throughput for RNNs.

[[slides]](Lecture 6 - Nvidia RNNs and GPUs.pdf)
[[video]](https://ox.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=f53e1dee-4985-4b9c-b230-6667b9da4ec0)

### Reading
* [Optimizing Performance of Recurrent Neural Networks on GPUs. Appleyard et al., arXiv 2016.](https://arxiv.org/abs/1604.01946)
* [Persistent RNNs: Stashing Recurrent Weights On-Chip, Diamos et al., ICML 2016](http://jmlr.org/proceedings/papers/v48/diamos16.pdf)
* [Efficient softmax approximation for GPUs. Grave et al., arXiv 2016.](https://arxiv.org/abs/1609.04309)


# Piazza
We will be using Piazza to facilitate class discussion during the course. Rather than emailing questions directly, I encourage you to post your questions on Piazza to be answered by your fellow students, instructors, and lecturers. However do please do note that all the lecturers for this course are volunteering their time and may not always be available to give a reponse.

Find our class page at: https://piazza.com/ox.ac.uk/winter2017/dnlpht2017/home

# Assessment
The primary assessment for this course will be a take-home assignment issued at the end of the term. This assignment will ask questions drawing on the concepts and models discussed in the course, as well as from selected research publications. The nature of the questions will include analysing mathematical descriptions of models and proposing extensions, improvements, or evaluations to such models. The assignment may also ask students to read specific research publications and discuss their proposed algorithms in the context of the course. In answering questions students will be expected to both present coherent written arguments and use appropriate mathematical formulae, and possibly pseudo-code, to illustrate answers.

The practical component of the course will be assessed in the usual way.

# Acknowledgements
This course would not have been possible without the support of [DeepMind](http://www.deepmind.com), [The University of Oxford Department of Computer Science](http://www.cs.ox.ac.uk/), [Nvidia](http://www.nvidia.com), and the generous donation of GPU resources from [Microsoft Azure](https://azure.microsoft.com).
