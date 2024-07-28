## --- Project Background ---

This is a short project which required us to build a software that takes and recognizes user drawn digits 0-9. We were free to use any pre-existing models to predict 
the user drawn digit.
We created the GUI using PyQt5 and the machine learning was handled by PyTorch.

## --- Installation instructions ---

Step 1: download and install miniconda from https://docs.conda.io/en/latest/miniconda.html. In this project we used python 3.8.

Step 2: open miniconda and type the following: **conda create –n py38 python=3.8** (when prompted enter **yes/y** to install packages).

Step 3: once the packages are installed, type the command on your screen to activate the new environment (in our case it was **conda activate py38**). Your environment 
should now have changed from (base) -> (py38).

Step 4: enter the following command to install pytorch **pip install PyQt5 torch torchvision**

Step 5: enter the following command to install matplotlib **pip install matplotlib**. (At this point you can type **python** followed by **import matplotlib** and this can
help avoid ModuleNotFoundError later on).

Step 6: download the **ui.py** AND **ai_with_class.py** files.

Step 7: download and install VS Code from https://code.visualstudio.com/. Next, install the [python package](https://imgur.com/Lc3SH2n) in VS Code.

NOTE: for more detailed versions of all dependencies used, please refer to the requirements.txt file for versions.

## --- Versions of packages ---

Version as of April. 6, 2021:
- Added a trained AI model
- Basic drawable canvas implemented.

Version as of April. 7, 2021:
- Added the ability to clear the canvas.
- GUI now opens centred on monitors.

Version as of April. 9, 2021:
- AI correctly compiles in the ui.py file and is able to correctly predict numbers 0, 2, 3, 4, 5, 6, 8 and 9.

Version as of April. 12, 2021:
- GUI overhaul.

Version as of April. 14, 2021:
- General GUI updates
- Added the prediction result in the GUI and a probability display for digit recognition.

Version as of April. 15, 2021:
- Further GUI clean up.
- Software now has the ability to train the model from the GUI.

Version as of April. 16, 2021: 
- Added a scrollable, pop up window for displaying images and implemented the MNIST dataset display.
- The software will now give pop up warnings for particular errors, e.g. trying to recognize a digit without a model being trained.
- Redesigned the way the digits were being recognized to ensure that the digit was centred on the tensor correctly.
- Added the ability to choose which model is being used to recognize the digit.

Version as of May. 1, 2021:
- General code clean up and commenting.
## --- Running the program ---

Step 1: ensure that the files are in the same directory and open them in VS Code. Ensure that the python interpreter you have is set to **['py38: conda](https://imgur.com/14tc9ob)**  and run the ui.py file by pressing the green arrow in the top right or by pressing F5.

Step 2: when you first run the program, you will get a pop up informing you that you must train a model before trying to recognize any numbers you draw. Click **Ok** on the
pop up and that should bring up a GUI that looks like [this](https://imgur.com/3efkMQJ). In its current itteration, only one model exists to predict the digit and that is 
**Model 1**. You can set the model in the scroll button in the bottom right where it says Model 1 in the GUI image linked. For now, leave it on **Model 1** and go file ->
Train Model 1. This will train the model and can take a few minutes to run. The window will be unresponsive during this time. A pop up will appear once the model is trained.
Click **Ok**.

Step 3: you can now draw on the canvas and click recognize or press Ctrl+R to recognize the digit you have drawn. To clear the window you can simply click the clear button or
press Ctrl+C. The prediction will appear in the green box in the top right. The certainty of the software is displayed as a horizontal bar chart on the right hand side.

Step 4: if future versions have models added, you must train each model before using them to predict a digit, however, once you have trained a model, it does NOT need to be 
retrained.

Step 5: you can view the MNIST datasets under the view menu tab.
