# DeepKeys
<h1>DeepKeys</h1> 
<h2>DeepKeys is a basic AI built to compose classical music in the styles of Mozart, Chopin, and Beethoven.</h2>

It uses a Recurrent Neural Network (RNN) to perform deep learning by listening to classical compositions by Mozart, Beethoven, and Chopin in order to generate authentic, classical compositons in the style of the three composers. The Keras API was used in developing the network model, using TensorFlow as the backend. Training is currently performed on 11 pieces (2 Beethoven, 3 Chopin, 6 Mozart). The training process (500 epochs) takes approximately 10 hours on a CPU (specs below).

<h3>CPU Specs:</h3>
<ul>
<li>Dell XPS 7500 Desktop</li>
<li>Intel (R) Core (TM) i7-3770 @ 3.40GHz</li>
<li>12.0 GB RAM</li>
<li>64-bit OS, x64-based processor</li>
</ul>

<h3>RNN properties:</h3>
<ul>
<li>2 LSTM (Long-Short Term Memory) layers w/ Dropout between ([0.2, 0.5] respectively)</li>
<li>1 FC (Fully-Connected) layer</li>
<li>Softmax Activation</li>
<li>Categorical Cross-Entropy loss function</li>
<li>RMSProp optimizer (learning rate = 0.001)</li>
</ul>

There is much work to be done, but features include:
<ul>
<li>A LSTM (Long-Short Term Memory) RNN model with two LSTM layers, Dropout,
        and a single fully connected layer.</li>
<li>A .mid file parser which decomposes a midi file into note data</li>
<li>A <code>predict</code> function which uses neural net predictions to stitch notes together to form a composition.</li>
</ul>

Short-term plans:
<ul>
<li>Experiment with network model (change hyperparameters, learning rate, layer structure, etc.)</li>
<li>Implement weight saving in order to facilitate training and keep track of training checkpoints</li>
</ul>
