# Description
 This machine learning model aimes to predict a person's salary based on their country, years of experience and level of education. The data was taken from [Stack Overflow's Annual Developer Survey of 2021](https://insights.stackoverflow.com/survey). The CSV data was taken and transformed using Pandas, a Data Analysis Library, to create a usable dataset which can be trained and tested. The model was saved using Pickle and used iinto the Streamlit front-end machine learning framework to be displayed.

# Instructions
 ## Machine Learning Model
  1. Install Jupyter Notebook [here](https://jupyter.org/install).
  2. Clone this repository to your computer.
  3. Open command prompt (or terminal for you Linux nerds) and navigate to the project directory.
  4. Run the command <code> $ jupyter notebook</code>.
  5. Open **ML Programming.ipynb**.
  6. Click on __Kernel__ > __Restart and & Run All__.

 ## Streamlit
  The website is already deployed [here](https://share.streamlit.io/moosahassanx/salary-predictor/main/app.py) but you can also locally run the web application:
   1. Open command prompt (or terminal for you Linux nerds again) and navigate to the project directory.
   2. Run the command <code> $ streamlit run app.py</code>.
   3. Your browser should automatically open a window/tab displaying the page.

# Credits
This code was inspired by [Patrick Loeber](https://www.python-engineer.com/) in which he uses dataset from 2020 in his ML model. My model is updated for the 2021 dataset. Other prediction models of the same dataset will be added to this project.