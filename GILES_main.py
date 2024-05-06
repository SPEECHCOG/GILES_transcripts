#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:49:02 2022

@author: rasaneno
"""

"""
Scripts used for Generator of Infant Language Experiences (GILES) 
child-diretected speech (CDS) text generation. Requires CHILDES_AO dataset, 
as organized into age-dependent bins of .txt files per corpus.
"""




import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
#import string
import datetime
import tensorflow_text as text
from os.path import exists
import pickle
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

os.environ["CUDA_VISIBLE_DEVICES"]="1"  # 0 or 1 for GPU selection


# Setup/experiment name definitions
model_basename = 'GILES_CHILDES_agecond_test' # name of the "experiment", used for saving the models/results
txt_out_name = 'agetests_test'  # nametag to add to each of the generated .txt files


# Data preparation settings
vocab_size = 8000   # Target vocabulary size for BERT word embedding training
maxlen = 100        # Sequence length for model input
dev_age_bin = 57    # CHILDES age bin to use for model validation

# LM hyperparameters
embed_dim = 512  # Embedding size for each token
num_heads = 8  # Number of attention heads
feed_forward_dim = 512  # Hidden layer size in feed forward network inside transformer
sampling_k = 500
n_layers = 5
dropout_rate = 0.05

# Training settings
batch_size=64
patience=15 
max_epochs = 1000   
train_model = 0     # 1: train from scratch, 0: load weights from file (based on model_basename)


# Generation settings 
samples_to_generate = 2000              # how many independent utterance sequences to generate
tokens_to_generate_per_sampling = 60    # number of tokens per generated sequence
ages_to_synth = [6,9,12,15,18,21,24,36,48] # age bins to synthesize
sampling_temperature = 1                # default = 1. Higher means more varied generation.
use_ageseed = 0                         # Sample text using age-bin dependent seed (0/1)?
seed_maxlength = 4                      # Max length of seed prompt used to start generation


# Specify locations for experimental scripts (maindir), location of AO-CHILDES (datadir) and logging (log-dir)

if(os.path.isdir('/Users/rasaneno/')):
    maindir = '/Users/rasaneno/Documents/koodit/dev/GILES_alpha/'       # Location of the code (where to save models etc.)
    datadir = "/Users/rasaneno/speechdb/aochildes_2021/CHILDES_ao_dataset_1word/"     # Where CHILDES_AO .txt files are located?
    log_dir = "/Users/rasaneno/rundata/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # Where to log training?
else:
    maindir = '/home/rasaneno/tools/GILES_alpha/'
    datadir = "/home/rasaneno/data/CHILDES_ao_dataset_1word/"
    log_dir = "/home/rasaneno/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    


#-----------------------------------------------------------------------------#
#---------------------FUNCTION DEFINITIONS START HERE-------------------------#
    

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)
    

def create_dynamic_padding_mask(x):
    mask = tf.math.equal(x, 0)  # Assuming 0 is the padding token
    mask = 1-tf.cast(mask, dtype=tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]



class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
    
        # Assuming inputs are token IDs; adjust if they are embeddings
        token_ids = tf.cast(inputs, dtype=tf.int32)[:, :, 0]  # Take the first feature as token id if the input is embeddings        
        # Create causal mask
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.float32)        
        # Create dynamic padding mask
        padding_mask = create_dynamic_padding_mask(token_ids)        
        # Use an outer product to create a 2D mask
        padding_mask_2d = tf.matmul(padding_mask, padding_mask, transpose_a=True)        
        # Make sure the padding_mask_2d shape is compatible with causal_mask
        padding_mask_2d = tf.reshape(padding_mask_2d, [batch_size, 1, seq_len, seq_len])        
        # Combine masks
        combined_mask = tf.minimum(causal_mask, padding_mask_2d)
        combined_mask = combined_mask > 0        
        # Proceed as before
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
        
    
def masked_categorical_crossentropy(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask  # apply the mask to zero-out the loss on padded tokens
    return tf.reduce_mean(loss)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    
# Define model structure

def create_model(feed_forward_dim=feed_forward_dim,n_layers=n_layers):
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    inputs2 = layers.Input(shape=(1,), dtype=tf.float32)
    inputs2 = keras.layers.Reshape((1,1))(inputs2)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    embedding_layer2 = keras.layers.Dense(embed_dim,activation='relu')
    dropout = layers.Dropout(dropout_rate)
    x = embedding_layer(inputs)
    x2 = embedding_layer2(inputs2)        
    x = tf.keras.layers.Concatenate(axis=-2)([x2,x])
    # Initialize an empty list to store the transformer blocks
    transformer_blocks = []    
    for i in range(0, n_layers):
        new_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
        transformer_blocks.append(new_block)
        x = transformer_blocks[-1](x)
        x = dropout(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=[inputs2,inputs], outputs=[outputs[:,1:,:], x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer, loss=[loss_fn, None])  
    return model



def write_vocab_file(filepath, vocab):
  with open(filepath, 'w') as f:
    for token in vocab:
      print(token, file=f)



def write_output_file(filepath, txt):
    f = open(filepath,'a')
    f.write(txt)
    f.write('\n')
    f.close()


def read_vocab_file(filepath):
    vocab = []
    with open(filepath, 'r') as f:
      for line in f:
        vocab.append(line.replace('\n',''))
    return vocab


def find_indices(lst, value):
    return [index for index, elem in enumerate(lst) if elem == value]


def remove_by_indices(original_list, indices_to_remove):
    # Ensure the indices are in descending order
    sorted_indices = np.sort(indices_to_remove)[::-1]
    # Remove the elements
    for index in sorted_indices:
        original_list.pop(index)
    return original_list


def prepare_lm_inputs_labels(text):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    tmp = en_tokenizer.tokenize(text)
    tmp = tmp.merge_dims(-2,-1)
    tmp = tmp.to_tensor(shape=[1,maxlen+1])
    tokenized_sentences = tmp
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y



def format_training_data(text_ds,ages,en_tokenizer,maxlen):        
    
    tmp = en_tokenizer.tokenize(text_ds)
    X = np.concatenate(np.concatenate(tmp.numpy()).ravel())
    Y = X[1:]
    X = X[:-1]
    # Convert the ages list to a tensor if it isn't one already
    ages_tensor = tf.constant(ages)
    
    # This will hold all the values for the new 3D ragged tensor
    flat_values = []
    
    # Iterate over each row in the original ragged tensor
    for i in range(len(ages)):
        # Get the lengths of the sublists for each 2D ragged tensor in the current row
        sublist_lengths = tmp[i].row_lengths()
    
        # Now we create a flat tensor of the age value, repeated according to the total length of the sublists
        repeated_values = tf.repeat(ages_tensor[i], tf.reduce_sum(sublist_lengths))
        
        # Add the repeated values to the list of flat values
        flat_values.append(repeated_values)
    
    # Use tf.stack to combine all scalar tensors into a single tensor
    flat_values_tensor = tf.concat(flat_values, axis=0)
    
    # Use the original nested row splits to create the new ragged tensor
    new_ragged_tensor = tf.RaggedTensor.from_nested_row_splits(flat_values_tensor, tmp.nested_row_splits)
    
    A = np.concatenate(np.concatenate(new_ragged_tensor.numpy()).ravel())
    A = A[:-1]
       
    # Reshape to batches

    n_batches = int(np.floor(len(Y)/maxlen))
    Y = np.reshape(Y[0:n_batches*maxlen],(n_batches,maxlen))    
    X = np.reshape(X[0:n_batches*maxlen],(n_batches,maxlen))
    A = np.reshape(A[0:n_batches*maxlen],(n_batches,maxlen))    
    
    # Prepare conditioning inputs 
    
    C = np.mean(A,axis=-1)
    C = np.reshape(C,(len(C),1))

    return X,Y,C


# Tensorflow Callbacks for model training and text generation

class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the tokenizer
        en_tokenizer: tokenizer for the data
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """
    def __init__(
        self, max_tokens, start_tokens, index_to_word, en_tokenizer, top_k=sampling_k, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k
        self.en_tokenizer = en_tokenizer
    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)
    def detokenize(self, number):
        return self.index_to_word[number]
    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            x = x.reshape(1, -1)  
            # Reshape 'second_input' to have shape (1, 1), making it a batch of size 1
            second_input = np.array([3]).reshape(1, 1)  
            y, _ = model.predict([second_input,x])
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        tokens_tensor = tf.convert_to_tensor(tokens_generated)
        tokens_tensor = tf.expand_dims(tokens_tensor,axis=-1)
        words = self.en_tokenizer.detokenize(tokens_tensor)
        txt = tf.strings.reduce_join(words, separator=' ', axis=-1)
        txt = txt.numpy()
        # Clean up formatting of the string
        txt = ((np.array2string(txt)).replace('b\'','')).replace('\'','').replace('b""','\'').replace('\n','').replace(' \' ','\'').replace(' ##','').replace('##','')
        print(f"generated text:\n{txt}\n")


# Sampling from posterior over the vocabulary
def sample_from(logits,temperature=1):
    logits, indices = tf.math.top_k(logits, k=sampling_k, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    preds = preds**(1/temperature)
    preds = preds/np.sum(preds)
    return np.random.choice(indices, p=preds)


# Text generation by sampling from the model after training
def sample_new(start_tokens,max_tokens,en_tokenizer,temperature=1,age_cond=5):
    start_tokens = [_ for _ in start_tokens]
    num_tokens_generated = 0
    tokens_generated = []
    while num_tokens_generated <= max_tokens:
        pad_len = maxlen - len(start_tokens)
        sample_index = len(start_tokens) - 1
        if pad_len < 0:
            x = start_tokens[:maxlen]
            sample_index = maxlen - 1
        elif pad_len > 0:
            x = start_tokens + [0] * pad_len
        else:
            x = start_tokens
        x = np.array([x])
        x = x.reshape(1, -1)  
        # Reshape 'second_input' to have shape (1, 1), making it a batch of size 1
        second_input = np.array([age_cond]).reshape(1, 1)  
        y, _ = model.predict([second_input,x])
        sample_token = sample_from(y[0][sample_index],temperature)
        tokens_generated.append(sample_token)
        start_tokens.append(sample_token)
        num_tokens_generated = len(tokens_generated)
    tokens_tensor = tf.convert_to_tensor(tokens_generated)
    tokens_tensor = tf.expand_dims(tokens_tensor,axis=-1)
    words = en_tokenizer.detokenize(tokens_tensor)
    txt = tf.strings.reduce_join(words, separator=' ', axis=-1)
    txt = txt.numpy()
    txt = ((np.array2string(txt)).replace('b\'','')).replace('\'','').replace('b""','\'').replace('\n','').replace(' \' ','\'')
    new_txt = txt
    txt = new_txt.replace('b\'','').replace('b""','\'').replace('\n','').replace(' \' ','\'').replace(' .','.').replace('b"','').replace("[. ",'').replace(']','').replace('[','').replace('_','').replace('  ',' ').replace('www','').replace(' ##','').replace('##','')
    txt = txt[txt.find('.')+2:] # remove before first period 
    txt = txt[0:txt.rfind('.')+1] # remove after last period
    print(f"generated text:\n{txt}\n")
    return txt
    
 
#-----------------------------------------------------------------------------#
#---------------------PIPELINE EXECUTION STARTS HERE--------------------------#


 
# Read training data from the folder
ages = []
file_names = []

# Recursively walk through all directories and sub-directories starting from the root directory
for root, dirs, files in os.walk(datadir):
    for file in files:
        if file.endswith(".txt"):
            # Add full file path to file_names list
            
            # Remove the file extension and split the filename
            #name_parts = os.path.splitext(file)[0].split("-")
            if root[-3:] != 'unk':
                file_names.append(os.path.join(root, file))
                ages.append(int(root[-3:]))
          

# Isolate chosen age bin for model validation
tmp = np.array(find_indices(ages,dev_age_bin))
filenames_dev = [file_names[i] for i in tmp]
ages_dev = [ages[i] for i in tmp]

# Remove the validation bin from data to create a training set
filenames_train = remove_by_indices(file_names,tmp)
ages_train = remove_by_indices(ages,tmp)


print(f"{len(filenames_train)} files for training")
print(f"{len(filenames_dev)} files for development")


# Create Tensorflow Datasets for data tokenization

content_list = []
for file_name in filenames_train:
    #file_path = os.path.join(folder_path, file_name)
    file_path = file_name
    with open(file_path, 'r') as f:
        content = f.read()
    content_list.append(content)

text_ds = np.array(content_list)


content_list = []
for file_name in filenames_dev:
    #file_path = os.path.join(folder_path, file_name)
    file_path = file_name
    with open(file_path, 'r') as f:
        content = f.read()
    content_list.append(content)

text_ds_dev = np.array(content_list)


# Tokenize training data
bert_tokenizer_params=dict(lower_case=True)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size = vocab_size,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)



vocab_pathname = maindir + 'token_vocabularies/en_vocab_ds_{}.txt'.format(vocab_size)
if(exists(vocab_pathname)):
    vocab = read_vocab_file(vocab_pathname)
else:
    text_ds_vocab = tf.data.Dataset.from_tensor_slices(text_ds)
    vocab = bert_vocab.bert_vocab_from_dataset(
        text_ds_vocab.batch(1000).prefetch(2),
        **bert_vocab_args
    )
    write_vocab_file(vocab_pathname, vocab)


# Print some examples of vocabulary
print(vocab[:10])
print(vocab[100:110])
print(vocab[1000:1010])
print(vocab[-10:])


en_tokenizer = text.BertTokenizer(vocab_pathname, **bert_tokenizer_params)



# Tokenize data and convert back to numpy arrays 
# (because, for some reason, it works better than training with TF datasets)

X_dev,Y_dev,C_dev = format_training_data(text_ds_dev,ages_dev,en_tokenizer,maxlen)
X_train,Y_train,C_train = format_training_data(text_ds,ages_train,en_tokenizer,maxlen)


#-----------------------------------------------------------------------------#
#---------------------------MODEL TRAINING------------------------------------#

# Create a text generation callback with a fixed start prompt to monitor trainijng progress
word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

start_prompt = "in the world of"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
text_gen_callback = TextGenerator(tokens_to_generate_per_sampling, start_tokens, vocab, en_tokenizer)



# Define a unique name for the model
modelname = model_basename + '_' + '{}layers_'.format(n_layers) + '{}emb_'.format(embed_dim) + '{}ff_'.format(feed_forward_dim) + '{}head_'.format(num_heads) + '{}voc_'.format(vocab_size) + '{}do'.format(int(dropout_rate*100))

checkpoint_filepath = maindir + '/checkpoint/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

log_dir = log_dir + modelname
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)


## Create model and train

model = create_model()


if(train_model):
    history = model.fit([C_train,X_train], Y_train, validation_data=[[C_dev,X_dev],Y_dev], batch_size=batch_size, shuffle=True, verbose=1, epochs=max_epochs, callbacks=[text_gen_callback,tensorboard_callback,earlystop_callback,model_checkpoint_callback])
    
    with open(maindir + 'models/' + modelname + '_history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    model.save_weights(maindir + 'models/' + modelname + '_w', overwrite=True, save_format=None, options=None)
    tf.keras.backend.clear_session()
else: # Load weights from previous training with the same model name
    model.load_weights(maindir + 'models/' + modelname + '_w')    

 
#-----------------------------------------------------------------------------#
#--------------------------TEXT GENERATION------------------------------------#


txt_out_dir = maindir + '/txt_out/' + txt_out_name  + '/' 
mypath=os.path.dirname(txt_out_dir) 
if not os.path.exists(txt_out_dir): 
  os.makedirs(txt_out_dir) 


# Generate text from the model by sampling while conditioning with the pre-defined age bins (ages_to_synth):
for age_cond in ages_to_synth:    
    tf.keras.backend.clear_session()
    if(use_ageseed):
        valid_inds,b = np.where(C_train == age_cond)
    else:
        valid_inds = np.arange(0,len(C_train))
    for r in range(0,samples_to_generate):        
        element = X_train[valid_inds[np.random.randint(len(valid_inds))],:]
        start_length = np.random.randint(1,seed_maxlength+1)
        i = np.random.randint(0,len(element)-start_length)
        start_tokens = np.array(element, dtype=np.int32)
        txt = sample_new(start_tokens[i:i+start_length].tolist(),tokens_to_generate_per_sampling,en_tokenizer,sampling_temperature,np.array(age_cond))
        write_output_file(maindir + '/txt_out/' + txt_out_name  + '/' + modelname + '_{}k'.format(sampling_k) + '_{}t'.format(int(sampling_temperature*100)) + '_{:03d}'.format(age_cond) + '.txt', txt)
        if(np.mod(r,200) == 0):
            tf.keras.backend.clear_session()
            
    

