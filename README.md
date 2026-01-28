<h1>Convolution Neural Network (CNN) Mushroom Identifier</h1>

The following project creates a fine-tuned CNN AI with MobileNetV2

The AI can be trained on any data set provided, although slight modifications 
<ol>
  <li>Navigate to Code/MyCode</li>
  <li>Plug train folder and validation folder directory into lines 7 & 8 </li>
  <p>IF YOUR DATASET IS NOT SET UP CORRECTLY IT WILL NOT WORK. IF IT IS NOT FORMATTED CORRECTLY PLUG IT INTO THE 'BuildDataset.py' FILE</p>
  <li>Change line 22 "predictLayer = Dense([# of classes i.e 3], activation='softmax')(x)" to correspond the amount of classes your dataset has</li>
</ol>

The 'oldXceptionBinaryClassifier' is a text file that contains a simple binary classifier, note that the neural network is written in the code from lines 79 and 144

Use the 'TestAI.py' to test your AI
Note: You will need to install the 'open-cv python' dependency to utilize the GUI display

Only do the previous if you need to train your own model, the final weights are provided.
Change the weights file to a '.keras' file, and test it in the testAI.py file 
