## Classifying emotions using audio recordings and Python

Emotion recognition technologies have been widely used in a variety of different applications in the past few years and
seem to be only growing more popular in today's industry; from psychological studies to advanced analysis of customer 
needs, it seems like the potential of these technologies is inexhaustible. Yet, these methods are still perceived as 
incredibly complex and inaccessible for the average data scientist who do not hold the privilege of being an expert in 
analysis of audio frames and sound waves.

In this article we'll aim at making this process as accessible and simplistic 
as we can by showing an example of an Emotion-Recognition classifier, using python and Librosa- a python package that 
makes the analysis of audio files incredibly easy and straight forward. We'll go through this process by explaining 
each step both theoretically and technically. 

First, in order to understand how to create an algorithm that processes 
audio sounds and understands their inherent meanings, we first need to understand how the human brain does the exact 
same thing.

### Sound waves and more
The first term we're going to talk about is the Sound Wave. 

Sound could be defined as a mechanical disturbance that propagates through an elastic material medium, such as air. 
It propagates in longitudinal waves (hence the term 'sound waves') and consists of alternating compressions 
and rare fractions, or regions of high pressure and low pressure, moving at a certain speed. The wayour bodies able to create 
sound waves originates in our Glottal Pulse, which manipulates the folds of our vocal cords when we're speaking. 
That is obviously very helpful for generating sounds but wouldn't be enough to actually make sense of them and 
communicate with each other. To accomplish that, we have our Vocal Tract.

The Vocal Tract is a term that describes a system of different organs in our mouth and throat- Nasal cavity, the tip of the tongue, teeth, soft palate and more. 
The goal of this system is to serve as the filter to our Glottal Pulse in a way that makes sense of the different sounds
we're generating. To make things easier, we can say that our speech is a result of the different movements and 
manipulations we're applying on our Glottal Pulse using our Vocal Tract. We'll talk more about the ways we can analyze 
these sound waves later, but for now let's keep these details in mind and dive into the more pragmatic part 
of this task.

### Our dataset - exploration of sound waves

To create this algorithm, we combined three different datasets which included 
voice recordings and their respective labels- Happy, Sad, Angry, Calm etc. The datasets we used are RAVDESS, TESS and
SAVEE (add links). Since the final dataset seemed to be pretty imbalanced toward some of the features (for example, 
we had much fewer male recordings then female recordings, and a relatively small number of 'positive' emotions compared 
to 'negative' ones) we decided to start with a simplified model first- classification of 3 different classes for 
both male and female (overall 6 different classes). The column 'emotion2' in our data frame served as our first
target column:

Using Librosa, which we mentioned earlier, we can plot the raw display of one of our sound waves: 
Now we would like to take this signal and display it in a time-frequency domain, so that we can examine the different 
frequencies and amplitudes of our signal over time. This is done by using the Fourier transformation on our data:
The plot above, which shows a linear representation of our frequencies over time, doesn’t seem to give us too much 
valuable information, and the reason for that is that the sounds human hear (unlike dogs, for example) are concentrated 
in a very small frequency and amplitude ranges. To fix that, we apply log-scale on both the frequency and the 
amplitudes. Note that now that we're applying log-scale on our data, we're no longer measuring our signalin units of 
amplitudes, but rather in units of decibels.  This result is much more informative as we can see the decibels of the 
different frequencies over time. The formal term for this representation is called Spectrogram.

### Diving deeper into the sound wave
Now that we saw spectrograms, let's consider the spectrum of our sound wave, that is, 
the representation of the Decibels against the Frequencies in our given time frame:

The log-scale Spectrum is achieved by applying Fourier transformation on our data, 
and then log-scale transformation on the result. The difference between Spectrograms 
and log-scale Spectrums, which are both being achieved by similar mathematical 
operations, is that while the first displays the frequencies and decibels over time, the latter shows the relation between the decibels and the 
frequencies; subtle difference but an important one.

Going back to our definition of speech, we can say that our 
log-scaled Spectrum is a pretty accurate representation of the speech itself. As we mentioned earlier, speech could be 
described as a combination of Vocal Tract and Glottal Pulse. In order to analyze ourspeech efficiently we would need 
to extract the Vocal Tract, which resembles a filter and includes the integral information regarding the meaning of the 
sounds, from the speech without the additional noise. To simplify things- we need to extract the Vocal Tract from the 
speech without the Glottal Pulse.Now let's consider a smoother version of our log-scaled Spectrum- the Spectral 
envelope:
The plot above resembles the main and most important parts of our log-scaled Spectrum- we can examine the main 
periodicity, the maximum points (marked with red), and the general trend as well. This Spectral envelope is the 
equivalent to what we considered as the Vocal Tract, and its maximum points are called 'Formants'- these contain the 
main 'identity' of the sound and considered as themost important parts of our sound wave. Research shows that our brain 
can identify most of the meaning of the speech just by its Formants.MFCC's for the missionNow that we recognized our 
Vocal Tract, we need to find a way to extract it from our speech. For that we're going to need to perform just a few 
more mathematical operations. First, we'll need to transform our log-scaled Spectrum to a Mel-scale1 and then perform 
a Discrete Cosine transform on our data. The last transformation is relatively similar to the Fourier transform 
we used before, as it also creates a Spectrum from our current state. So basically, what we're doing here is creating 
a Spectrum on our previous Spectrum. Our new Spectrum is now called Cepstrum, while our new frequencies are called 
Quefrencies. The reason for all that confusion is pretty simple- using our new Cepstrum and Quefrencies we can 
differentiate easily between our Spectral envelope (the Vocal Tract) and the noise (the Glottal Pulse). 
More then that, the Discrete Cosine transformation already results in the main and most 1a scale that resembles 
humans sound spectrum in a more realistic manner, given the factthat the human ear cannot identify changes in 
frequencies higher then 500Hz very easily.
important coefficients of this cepstrum- the Mel-frequency cepstrumcoefficients, also known as MFCC's! Lucky for us, 
Librosa makes the job incredibly easy, and we can extract the MFCC's in one simple line of code, as we'll soon see.
Data preparationTo use the MFCC's as our features we first needed them to be in a homogenous shape. 
Since each one of our signal waves is of slightlydifferent length, the result would have led to MFCC's of different 
lengths as well, which won't be very helpful in creating a model for our data. Therefore, we started by making all the
signal waves the same length- signals shorter than the mean were padded (using the median) and signals longer than the 
mean were truncated (we also took down the ones with extreme lengths beforehand). Afterwards we extracted the MFCC's 
easily from each signal wave. Each signal wave had 20 different coefficients, and each coefficient included a vector 
of the given length. Note- The traditional number of MFCC's is usually around 12-13, and the default in the Librosa 
function is 20.As we can see, each sample includes 20 MFCC's of length 236 each. We also dropped 3 outliers 
along the way. 
Now we can finally start modeling.ModelingRNN networks are known to work well for speech recognition tasks. 
however, there's a strong body of research that proves that CNN networks can outperform RNN networks in a lot of cases. 
In this case, we decided to go for CNN networks using 1-dimensional convolution layers and 1-dimensional pooling layers 
(as our training data is made of 3 dimensions).Our CNN network consisted of two blocks, each built of a 1-dimensional 
convolution layer, activation function ('ReLu'), 1-dimensional pooling layer and dropout. The blocks were followed 
by two fully connected dense layers and a 'SoftMax' activation function,as we are dealing with a multi-class task:
We managed to reach relatively good results, reaching 90% accuracy. As expected, the scores of the male recordings 
were significantly lower than the female ones, as we had much fewer male samples in our dataset:After that we aimed at 
a more complex model- classifying different emotions- 'fear', 'surprise', 'sadness', 'negative', 'positive' and 
'neutral'2. The distribution of the samples was more balanced than 2These emotions appear in column 'emotion3' 
on the data frame. Note that we used the emotions that had more samples in order to reach a balanced classification.
the one we had before, hence we expected a more balanced classification this time.We applied the same processing to 
the data as we have before and ran the exact same model. These are the results we reached:As expected, this 
classification seems relatively balanced, and with pretty good results as well. We can observe the Confusion 
matrix as well, which shows that the vast majority of our samples were classified correctly:
We later wanted to examine the relations between these classes a bit further, so we used KMeans clustering with 
6 different clusters, and performed dimensionality reduction over our dataset using PCA3. That allowed us to 
visualize our samples and examine the 'distances' between the different classes: We can see that 'surprise', 'positive' 
and 'neutral' are pretty close, as well as 'fear' and 'negative'. These relations make sense as 'fear' can very well be 
defined as a negative emotion, and the proximity between 'surprise' and 'positive' seems relatively intuitive as well. 

3We only used the first two dimensions of our dataset, as our dataset's dimension (which had the shape of (994, 236, 20)
was too big to perform these operations on them.
The position of 'sadness' seems relatively vague opposed to the other emotions; this could be explained by the fact 
that this class included much less samples compared to other classes, which let to worse predictions. 
In an ideal model we would expect to see it closerto 'negative' and 'fear'.Overall, these results prove that even 
with a simple CNN network we can classify emotions from audio recordings with high levels of confidence.Final words
What makes audio classification so interesting is the way complex theory could be expressed by common practical means. 
The way mathematical procedures can be used to successfully implement these concepts is nothing less than remarkable. 
Lucky for us, most of the hard work is being done by pre-made tools; however, knowingthe theoretical concepts that 
lies behind these tools remains the keyaspect in building and using these models successfully. 

We hope that this article helped by shedding some light on this fascinating subject and making it a little more 
apprehensible, and hopefully more usable as well. 

References
* https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/CNN_ASLPTrans2-14.pdf
* https://www.izotope.com/en/learn/understanding-spectrograms.html
* https://www.youtube.com/watch?v=4_SH2nfbQZ8&t=1357s-