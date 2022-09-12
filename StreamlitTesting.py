import streamlit as st
st.title('Welcome to Streamlit App!')

st.write('Hello, *World*, sunglasses')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

st.write(pd.DataFrame({'First Column' : [1, 2, 3, 4, 5], 'Second Column' : [10, 20, 30, 40, 50],}))
st.write('1 + 1 = ', 2)
st.header('This is a header')
st.subheader('This is subheader')
st.caption('This is a string that explains something above.')
code = '''def hello(): print("Hello, streamlit!")'''
st.code(code, language = 'python')
st.markdown('This is equation')
st.latex(r'''a + ar + ar^2 + ar^3 + \cdots + ar^{n-1} = \sum_{k=0}^{n-1}ar^k = a\left(\frac{1 - r^{n}}{1 - r}\right)''')

data_chart = pd.DataFrame(np.random.randn(20, 3), columns = ['a', 'b', 'c'])
st.title('Line Chart')
st.line_chart(data_chart)
st.write('Area Chart')
st.area_chart(data_chart)
st.text('Bar Chart')
st.bar_chart(data_chart)

array = np.random.normal(1, 1, size = 100)
fig, ax = plt.subplots()
ax.hist(array, bins = 20)
st.header('Histogram')
st.pyplot(fig)

uploaded_file = st.file_uploader('Upload your file here!')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.header('Data Statistics')
    st.write(df.describe())
    st.header('Data Header')
    st.write(df.head())
    
df = pd.DataFrame(np.random.randn(200, 3), columns = ['a', 'b', 'c'])
c = alt.Chart(df).mark_circle().encode(x = 'a', y = 'b', size = 'c', color = 'c', tooltip = ['a', 'b', 'c'])
st.title('ALT Chart')
st.write(c)
st.altair_chart(c, use_container_width = True)
    
col1, col2 = st.columns(2)
col1.write('This is column one.')
col2.write('This is column two.')   
    
df = pd.DataFrame(np.random.randn(50, 20), columns = ('col %d' %i for i in range(20)))
st.header('Table One')
st.dataframe(df)
st.title('Table')
st.write(df) 

df = pd.DataFrame(np.random.randn(10, 15), columns = ('col %d' %i for i in range(15)))
st.dataframe(df.style.highlight_max(axis = 0))

df = pd.DataFrame(np.random.randn(10, 5), columns = ('col %d' %i for i in range(5)))
st.table(df)

st.dataframe(df, 200, 100)

st.metric(label = 'Temperature', value = '70 F', delta = '1.2 F')

col1, col2, col3 = st.columns(3)
col1.metric('Temperature', '70 F', '1.2 F')
col2.metric('Wind', '9 mph', '-8 %')
col3.metric('Humidity', '86 %', '4 %')

st.metric(label = 'Gas Price', value = 4, delta = -0.5, delta_color = 'inverse')
st.metric(label = 'Active Developers', value = 123, delta = 123, delta_color = 'off')
    
st.json({'foo' : 'bar', 'baz' : 'boz', 'stuff' : ['stuff 1', 'stuff 2', 'stuff 3', 'stuff 5', ], })  

if st.button('Say Hello'):
    st.write('Why hello there!')
else:
    st.write('Goodbye')
    
text_contents = '''This is some text'''
st.download_button('Download some text', text_contents)

binary_contents = b'example content'
st.download_button('Download binary file', binary_contents)    
    
agree = st.checkbox('I agree')
if agree:
    st.write('Great')

genre = st.radio('What is your favourite movie genre?', ('Comedy', 'Drama', 'Documentary'))
if genre == 'Comedy':
    st.write('You selected comedy.')
else:
    st.write('You did not select comedy.')

option = st.selectbox('How would you like to be contacted?', ('Email', 'Home Phone', 'Mobile Phone'))
st.write('You selected ', option)

colors = st.multiselect('What are your favourite colors?', ['Green', 'Yellow', 'Red', 'Grey', 'Blue'], ['Yellow', 'Red'])
st.write('You selected ', colors)

title = st.text_input('Movie title', 'Life of Brain')
st.write('The current movie title is ', title)

number = st.number_input('Insert a number!')
st.write('The current number is ', number)

txt = st.text_area('Text to analyze!', '''How are you? I am good. And you? I hope you and your family will be healthy and wealthy.''')
st.write('Sentiment ', txt)

uploaded_file = st.file_uploader('Choose a file')
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)
    stringio = StringIO(uploaded_file.getvalue().describe('utf-8'))
    st.write(stringio)
    string_data = stringio.read()
    st.write(string_data)
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

uploaded_files = st.file_uploader('Choose a CSV file', accept_multiple_files = True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write('File name: ', uploaded_file.name)
    st.write(bytes_data)

color =st.color_picker('Pick A color', '#00f900')
st.write('The current color is ', color)  
    
    
    
    
    