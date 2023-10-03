# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.22947Z","iopub.execute_input":"2023-08-16T10:26:52.230126Z","iopub.status.idle":"2023-08-16T10:26:52.238394Z","shell.execute_reply.started":"2023-08-16T10:26:52.230066Z","shell.execute_reply":"2023-08-16T10:26:52.237424Z"}}
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras 
from tqdm import tqdm
from keras.layers import Dense
import json 
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
from sklearn.model_selection import train_test_split

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.240592Z","iopub.execute_input":"2023-08-16T10:26:52.241102Z","iopub.status.idle":"2023-08-16T10:26:52.2585Z","shell.execute_reply.started":"2023-08-16T10:26:52.24106Z","shell.execute_reply":"2023-08-16T10:26:52.25766Z"}}
question  =[]
answer = []
with open("../input/simple-dialogs-for-chatbot/dialogs.txt",'r') as f :
    for line in f :
        line  =  line.split('\t')
        question.append(line[0])
        answer.append(line[1])
print(len(question) == len(answer))

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.259982Z","iopub.execute_input":"2023-08-16T10:26:52.260669Z","iopub.status.idle":"2023-08-16T10:26:52.267042Z","shell.execute_reply.started":"2023-08-16T10:26:52.260635Z","shell.execute_reply":"2023-08-16T10:26:52.266104Z"}}
question[:5]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.269865Z","iopub.execute_input":"2023-08-16T10:26:52.270663Z","iopub.status.idle":"2023-08-16T10:26:52.277444Z","shell.execute_reply.started":"2023-08-16T10:26:52.270599Z","shell.execute_reply":"2023-08-16T10:26:52.276406Z"}}
answer[:5]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.278969Z","iopub.execute_input":"2023-08-16T10:26:52.28002Z","iopub.status.idle":"2023-08-16T10:26:52.294224Z","shell.execute_reply.started":"2023-08-16T10:26:52.279985Z","shell.execute_reply":"2023-08-16T10:26:52.293242Z"}}
answer = [ i.replace("\n","") for i in answer]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.295754Z","iopub.execute_input":"2023-08-16T10:26:52.296479Z","iopub.status.idle":"2023-08-16T10:26:52.305656Z","shell.execute_reply.started":"2023-08-16T10:26:52.296446Z","shell.execute_reply":"2023-08-16T10:26:52.304628Z"}}
answer[:5]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.307144Z","iopub.execute_input":"2023-08-16T10:26:52.308227Z","iopub.status.idle":"2023-08-16T10:26:52.322131Z","shell.execute_reply.started":"2023-08-16T10:26:52.308148Z","shell.execute_reply":"2023-08-16T10:26:52.320993Z"}}
data = pd.DataFrame({"question" : question ,"answer":answer})
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.323721Z","iopub.execute_input":"2023-08-16T10:26:52.324926Z","iopub.status.idle":"2023-08-16T10:26:52.330994Z","shell.execute_reply.started":"2023-08-16T10:26:52.324887Z","shell.execute_reply":"2023-08-16T10:26:52.329939Z"}}
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.332638Z","iopub.execute_input":"2023-08-16T10:26:52.333379Z","iopub.status.idle":"2023-08-16T10:26:52.346384Z","shell.execute_reply.started":"2023-08-16T10:26:52.333345Z","shell.execute_reply":"2023-08-16T10:26:52.345385Z"}}
def clean_text(text):
    text = unicode_to_ascii(text.lower().strip())
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = re.sub("(\\W)"," ",text) 
    text = re.sub('\S*\d\S*\s*','', text)
    text =  "<sos> " +  text + " <eos>"
    return text
    

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.435358Z","iopub.execute_input":"2023-08-16T10:26:52.438924Z","iopub.status.idle":"2023-08-16T10:26:52.44743Z","shell.execute_reply.started":"2023-08-16T10:26:52.438871Z","shell.execute_reply":"2023-08-16T10:26:52.446289Z"}}
data["question"][0]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.45351Z","iopub.execute_input":"2023-08-16T10:26:52.458475Z","iopub.status.idle":"2023-08-16T10:26:52.775763Z","shell.execute_reply.started":"2023-08-16T10:26:52.458378Z","shell.execute_reply":"2023-08-16T10:26:52.774693Z"}}
data["question"] = data.question.apply(clean_text)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.777083Z","iopub.execute_input":"2023-08-16T10:26:52.782932Z","iopub.status.idle":"2023-08-16T10:26:52.792522Z","shell.execute_reply.started":"2023-08-16T10:26:52.782895Z","shell.execute_reply":"2023-08-16T10:26:52.791616Z"}}
data["question"][0]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:52.796084Z","iopub.execute_input":"2023-08-16T10:26:52.796748Z","iopub.status.idle":"2023-08-16T10:26:53.099242Z","shell.execute_reply.started":"2023-08-16T10:26:52.796714Z","shell.execute_reply":"2023-08-16T10:26:53.098192Z"}}
data["answer"] = data.answer.apply(clean_text)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.109145Z","iopub.execute_input":"2023-08-16T10:26:53.109893Z","iopub.status.idle":"2023-08-16T10:26:53.117632Z","shell.execute_reply.started":"2023-08-16T10:26:53.109847Z","shell.execute_reply":"2023-08-16T10:26:53.116587Z"}}
question  = data.question.values.tolist()
answer =  data.answer.values.tolist()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.119306Z","iopub.execute_input":"2023-08-16T10:26:53.12004Z","iopub.status.idle":"2023-08-16T10:26:53.130256Z","shell.execute_reply.started":"2023-08-16T10:26:53.119997Z","shell.execute_reply":"2023-08-16T10:26:53.129263Z"}}
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, lang_tokenizer

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.132224Z","iopub.execute_input":"2023-08-16T10:26:53.13323Z","iopub.status.idle":"2023-08-16T10:26:53.284775Z","shell.execute_reply.started":"2023-08-16T10:26:53.133165Z","shell.execute_reply":"2023-08-16T10:26:53.283797Z"}}
input_tensor , inp_lang  =  tokenize(question)


# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.286131Z","iopub.execute_input":"2023-08-16T10:26:53.286801Z","iopub.status.idle":"2023-08-16T10:26:53.430856Z","shell.execute_reply.started":"2023-08-16T10:26:53.286765Z","shell.execute_reply":"2023-08-16T10:26:53.429852Z"}}
target_tensor , targ_lang  =  tokenize(answer)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.432185Z","iopub.execute_input":"2023-08-16T10:26:53.433091Z","iopub.status.idle":"2023-08-16T10:26:53.437493Z","shell.execute_reply.started":"2023-08-16T10:26:53.433053Z","shell.execute_reply":"2023-08-16T10:26:53.436441Z"}}
 #len(inp_question) ==  len(inp_answer)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.438961Z","iopub.execute_input":"2023-08-16T10:26:53.4396Z","iopub.status.idle":"2023-08-16T10:26:53.448507Z","shell.execute_reply.started":"2023-08-16T10:26:53.439556Z","shell.execute_reply":"2023-08-16T10:26:53.447447Z"}}
def remove_tags(sentence):
    return sentence.split("<start>")[-1].split("<end>")[0]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.450367Z","iopub.execute_input":"2023-08-16T10:26:53.451007Z","iopub.status.idle":"2023-08-16T10:26:53.46081Z","shell.execute_reply.started":"2023-08-16T10:26:53.450973Z","shell.execute_reply":"2023-08-16T10:26:53.459763Z"}}
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]


# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.462415Z","iopub.execute_input":"2023-08-16T10:26:53.463029Z","iopub.status.idle":"2023-08-16T10:26:53.473379Z","shell.execute_reply.started":"2023-08-16T10:26:53.462996Z","shell.execute_reply":"2023-08-16T10:26:53.472289Z"}}
# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.475887Z","iopub.execute_input":"2023-08-16T10:26:53.476813Z","iopub.status.idle":"2023-08-16T10:26:53.488978Z","shell.execute_reply.started":"2023-08-16T10:26:53.476775Z","shell.execute_reply":"2023-08-16T10:26:53.487947Z"}}
#print(len(train_inp) , len(val_inp) , len(train_target) , len(val_target))

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.495361Z","iopub.execute_input":"2023-08-16T10:26:53.495979Z","iopub.status.idle":"2023-08-16T10:26:53.547112Z","shell.execute_reply.started":"2023-08-16T10:26:53.495943Z","shell.execute_reply":"2023-08-16T10:26:53.546226Z"}}
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.550964Z","iopub.execute_input":"2023-08-16T10:26:53.55317Z","iopub.status.idle":"2023-08-16T10:26:53.564069Z","shell.execute_reply.started":"2023-08-16T10:26:53.553133Z","shell.execute_reply":"2023-08-16T10:26:53.563028Z"}}
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x,hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.565635Z","iopub.execute_input":"2023-08-16T10:26:53.566264Z","iopub.status.idle":"2023-08-16T10:26:53.613261Z","shell.execute_reply.started":"2023-08-16T10:26:53.566229Z","shell.execute_reply":"2023-08-16T10:26:53.612384Z"}}
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.617441Z","iopub.execute_input":"2023-08-16T10:26:53.619674Z","iopub.status.idle":"2023-08-16T10:26:53.632651Z","shell.execute_reply.started":"2023-08-16T10:26:53.619616Z","shell.execute_reply":"2023-08-16T10:26:53.631457Z"}}
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.638726Z","iopub.execute_input":"2023-08-16T10:26:53.641715Z","iopub.status.idle":"2023-08-16T10:26:53.681141Z","shell.execute_reply.started":"2023-08-16T10:26:53.641674Z","shell.execute_reply":"2023-08-16T10:26:53.680169Z"}}
attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.685412Z","iopub.execute_input":"2023-08-16T10:26:53.687793Z","iopub.status.idle":"2023-08-16T10:26:53.701891Z","shell.execute_reply.started":"2023-08-16T10:26:53.687752Z","shell.execute_reply":"2023-08-16T10:26:53.700638Z"}}
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.707251Z","iopub.execute_input":"2023-08-16T10:26:53.710125Z","iopub.status.idle":"2023-08-16T10:26:53.781955Z","shell.execute_reply.started":"2023-08-16T10:26:53.710086Z","shell.execute_reply":"2023-08-16T10:26:53.780959Z"}}
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.783222Z","iopub.execute_input":"2023-08-16T10:26:53.783809Z","iopub.status.idle":"2023-08-16T10:26:53.801117Z","shell.execute_reply.started":"2023-08-16T10:26:53.783772Z","shell.execute_reply":"2023-08-16T10:26:53.799988Z"}}
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.80313Z","iopub.execute_input":"2023-08-16T10:26:53.804135Z","iopub.status.idle":"2023-08-16T10:26:53.815038Z","shell.execute_reply.started":"2023-08-16T10:26:53.804097Z","shell.execute_reply":"2023-08-16T10:26:53.813898Z"}}
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<sos>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:26:53.817305Z","iopub.execute_input":"2023-08-16T10:26:53.818212Z","iopub.status.idle":"2023-08-16T10:30:49.723448Z","shell.execute_reply.started":"2023-08-16T10:26:53.818156Z","shell.execute_reply":"2023-08-16T10:30:49.722434Z"}}
EPOCHS = 40

for epoch in tqdm(range(1, EPOCHS + 1), desc='Epochs', unit='epoch'):
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

    if epoch % 4 == 0:
        print('Epoch:{:3d} Loss:{:.4f}'.format(epoch, total_loss / steps_per_epoch))


# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:30:49.725275Z","iopub.execute_input":"2023-08-16T10:30:49.725901Z","iopub.status.idle":"2023-08-16T10:30:49.736936Z","shell.execute_reply.started":"2023-08-16T10:30:49.725863Z","shell.execute_reply":"2023-08-16T10:30:49.735874Z"}}
def evaluate(sentence):
    sentence = clean_text(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<sos>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<eos>':
            return remove_tags(result), remove_tags(sentence)

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return remove_tags(result), remove_tags(sentence)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:30:49.738443Z","iopub.execute_input":"2023-08-16T10:30:49.738951Z","iopub.status.idle":"2023-08-16T10:30:49.759729Z","shell.execute_reply.started":"2023-08-16T10:30:49.738916Z","shell.execute_reply":"2023-08-16T10:30:49.758585Z"}}
questions  =[]
answers = []
with open("../input/simple-dialogs-for-chatbot/dialogs.txt",'r') as f :
    for line in f :
        line  =  line.split('\t')
        questions.append(line[0])
        answers.append(line[1])
print(len(question) == len(answer))

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:30:49.761114Z","iopub.execute_input":"2023-08-16T10:30:49.761651Z","iopub.status.idle":"2023-08-16T10:30:49.857691Z","shell.execute_reply.started":"2023-08-16T10:30:49.761592Z","shell.execute_reply":"2023-08-16T10:30:49.856684Z"}}
def ask(sentence):
    result, sentence = evaluate(sentence)

    print('Question: %s' % (sentence))
    print('Predicted answer: {}'.format(result))
ask(questions[100])

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:31:51.217121Z","iopub.execute_input":"2023-08-16T10:31:51.217506Z","iopub.status.idle":"2023-08-16T10:31:51.311879Z","shell.execute_reply.started":"2023-08-16T10:31:51.217474Z","shell.execute_reply":"2023-08-16T10:31:51.310786Z"}}
ask(questions[20])

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:32:02.970935Z","iopub.execute_input":"2023-08-16T10:32:02.971331Z","iopub.status.idle":"2023-08-16T10:32:02.977251Z","shell.execute_reply.started":"2023-08-16T10:32:02.971299Z","shell.execute_reply":"2023-08-16T10:32:02.976197Z"}}
print(answers[20])

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:32:25.21633Z","iopub.execute_input":"2023-08-16T10:32:25.216744Z","iopub.status.idle":"2023-08-16T10:32:25.263315Z","shell.execute_reply.started":"2023-08-16T10:32:25.216708Z","shell.execute_reply":"2023-08-16T10:32:25.262317Z"}}
ask(questions[10])

# %% [code] {"execution":{"iopub.status.busy":"2023-08-16T10:32:19.451059Z","iopub.execute_input":"2023-08-16T10:32:19.451469Z","iopub.status.idle":"2023-08-16T10:32:19.457131Z","shell.execute_reply.started":"2023-08-16T10:32:19.451436Z","shell.execute_reply":"2023-08-16T10:32:19.456008Z"}}
print(answers[10])

# %% [code]


# %% [code]


# %% [code]
