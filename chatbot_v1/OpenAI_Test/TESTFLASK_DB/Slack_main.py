import os,slack_sdk,json,requests
import mysql.connector
import timedelta
import ast

from datetime import datetime,date, timedelta


from flask import Flask, request, jsonify
from sqlalchemy import create_engine
from dotenv import load_dotenv
from threading import Thread
from mysql.connector import Error

from langchain_core.prompts import PromptTemplate
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_pinecone import PineconeVectorStore

from slackeventsapi import SlackEventAdapter

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Connect with both the application and the adapter

slack_event_adapter = SlackEventAdapter(os.getenv("SIGNING_SECRET"), '/slack/events', app)
client = slack_sdk.WebClient(token=os.getenv("SLACK_TOKEN"))



# Database Configuration
config = {
    'user': "test_chatbot_readonly",
    'password': os.getenv("PASS_USER"),
    'host': os.getenv("HOST"),
    'database': "test_1",
    'port': "3306"
}
cnx = mysql.connector.connect(**config)
cursor=cnx.cursor()


# SQLAlchemy Engine URI
pg_uri = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
engine = create_engine(pg_uri)
db = SQLDatabase(engine)

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
top_k=10
#need to change this
table_info_2=db.get_context()
table_info=db.get_table_info(['bookings','Users'])

print(table_info)

#might need to create a permanent vector db  and not a in-memory one



path_examples= r"CHATBOT\chatbot\test1chatbot_simple\OpenAI_Test\TESTFLASK_DB\Examples.txt"
#create a List of dictionaries from the Examples.txt file
with open(path_examples) as f:
    data = ast.literal_eval(f.read())

#using open-source sentence transformers ML models directly from huggingface
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

#create the example selector based on Facebook AI similarity search db
example_selector_faiss = SemanticSimilarityExampleSelector.from_examples(
    examples=data,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    embeddings=embeddings,
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    vectorstore_cls=FAISS,
    # The number of examples to produce Maximum is based on len(data)
    k=5,
)


example_prompt = PromptTemplate.from_template("User input: {input}\nSQL query: {query}")

#using prompts to have better quality output
prompt = FewShotPromptTemplate(
    example_selector=example_selector_faiss,
    example_prompt=example_prompt,
    prefix="You are a Mysql expert. Given an input question, create a syntactically correct MYSQL query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.Use always the test_1 database and use tables like this test_1.tablename to retrive data",
    suffix="User input: {input}\nSQL query: ",
    input_variables=["input", "top_k", "table_info"],
)

#creating the chain to connect all the configured parts inside the application
 
chain = create_sql_query_chain(llm, db, prompt) 

'''////////////////////////////////////////////////'''
#formatting for blocks inside slack,only used if the response is less than 50 blocks
def formatting(result, cursor):
    blocks = []
    num_results = len(result)  # Count the number of results

    for idx, row in enumerate(result):
        fields = []
        # Prepare fields for the Slack message
        fields.append({"type": "mrkdwn", "text": f"*Result ID*: {idx}"})
        for col_idx, col_description in enumerate(cursor.description):
            field_value = f"*{col_description[0]}*: {str(row[col_idx])}"
            fields.append({"type": "mrkdwn", "text": field_value})

        # Split fields into multiple sections if they exceed Slack's limit
        while fields:
            block = {
                "type": "section",
                "fields": fields[:10]
            }
            blocks.append(block)
            if len(fields) > 10:
                blocks.append({"type": "divider"})  # Adds a divider only if there are more fields to follow
            fields = fields[10:]  # Continue with the remaining fields, if any

        # Add a custom separator line after each result
        if idx < num_results - 1:  # Avoid adding separator after the last result
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"}})

    return blocks


#only used if there are more than 50 blocks in the response
#The result will be added inside the response.txt or response.html file
def formatting2(result,cursor):

    result_json = []

    idrow = -1
    for idrow, row in enumerate(result):
            row_dict = {"1ResultID": idrow}
            for idx, col in enumerate(cursor.description):
                    row_dict[col[0]] = str(row[idx])
                    #row_dict[col[0]] = row[idx]
            result_json.append(row_dict)

    return result_json 



'''Both dbinfoworker and promptworker are used after giving a ok response in order not to timeout.'''

#used to send database information directly inside slack 
def db_info_worker(response_url):
    #might change it with less information sent
    info=db.get_table_info_no_throw()

    payload = {"text": info, "response_type": "ephemeral"} #ephemeral=private message 
    try:
        requests.post(response_url, json=payload) #send request to response_url

    except requests.RequestException as e:
        print(f"Failed to send data: {e}")

#it will return the requested data from the user
def prompt_worker(response_url,text,user_name,channel):
    
    #security check,need to sanitize the text before sending it to the chain
    if ';' in text:
        warning={"text":"Do not use ;","response_type":"in_channel"}
        requests.post(response_url, json=warning,headers = {'Content-type': 'application/json'})

    #might need to change the max result
    sqlquery=chain.invoke({"question": f"{text}limit the result to 40"})

    try: 
        try:
            
            cursor.execute(sqlquery)
            result = cursor.fetchall() #return all executed data
    
            #formatting for better quality response
            result_json=formatting(result=result,cursor=cursor)
            blocks={"blocks": result_json,"response_type": "ephemeral"}

            if len(result) == 0:
                
                payloadnull={"text":"No data found","response_type":"ephemeral"}
                requests.post(response_url, json=payloadnull,headers = {'Content-type': 'application/json'})
            

            elif len(result_json) <= 50:    
                requests.post(response_url, json=blocks)
                
            #if the lenght of result_json(based on how many blocks are inside) is more than 50
            #send a file with the results as content
            elif len(result_json) > 50:
                   
                   result_json2=formatting2(result=result,cursor=cursor)
                   client.files_upload_v2(token=os.getenv("SLACK_TOKEN"),content=f"{result_json2}",filename="results.txt",channel=channel)
                   payload={"text":"Open this file to get the response","response_type":"ephemeral"}
                    
                   requests.post(response_url, json=payload,headers = {'Content-type': 'application/json'})
        
        except mysql.connector.errors.ProgrammingError as mysqlerror:
            return mysqlerror
        
        #logging all the prompts made by the user inisde the Logging_Chatbot table
        finally:    
            timestamp=datetime.now() 
            
            sqlqueryesc = sqlquery.replace('"', '\\"')  # replace =" into =\\"
            query=f'''INSERT INTO Logging_Chatbot (user,question,query,time)
            VALUES ('{user_name}','{text}',"{sqlqueryesc}",'{timestamp}')'''

            cursor.execute(query)
            cnx.commit() # save inside the database
               
            #examples.append([f"input:{text},",f"query:{query}"]) 
            #saves it into examples or external file 

            #:segno_spunta_bianco:    /   :x:

            #reactions=client.reactions_get(channel=channel,timestamp=today,)
            #reactions based on the last 5 messages,if the reply is bad you could flag it in the db
        '''
            if reaction=='check':
                examples.remove(last_line)
                examples.append({f"input:{text},",f"query:{query}"}) 

        '''
# Define the new example
        new_example = {"input": 3, "query": "Example text 3"}

# Append the new example to the data list
        data.append(new_example)
        
        with open(path_examples, 'w') as file:
           file.write(str(data))


    except Error as e:
      
      error_message = f"MySQL Error: {e.msg}"
      payloaderror ={"text": json.dumps(error_message), "response_type": "in_channel"}
      requests.post(response_url, json=payloaderror)
      return e.msg


        

'''////////////////////////////////////////////////////////////////'''
#starting the db info command
@app.route('/db_info', methods=['POST'])
def db_info_receptionist():
    user_name = data.get('user_name')
    response_url = request.form.get("response_url")
    if not response_url:
        return jsonify(text="No response URL provided."), 400

    #Send Ok at the user in order to not timeout the request inside slack
    response = jsonify(text=f"Give me a second ... {user_name}")
    response.status_code = 200
    response.headers['Content-Type'] = 'application/json'
    #Process in the background
    thr = Thread(target=db_info_worker, args=(response_url,))
    thr.start()

    return response



#starting the prompt command
@app.route('/prompt', methods=['POST'])
def prompt_info_receptionist():
    #retrieving necessary data
    data = request.form
    user_name = data.get('user_name')
    channel = data.get('channel_id')
    text = data.get('text')
    response_url = request.form.get("response_url")
   
    if not response_url:
            return jsonify(text="No response URL provided."), 400

    #Send ok 
    response = jsonify(text=f"Processing your request ... {user_name} wait a moment")
    response.status_code = 200
    response.headers['Content-Type'] = 'application/json'
    
    #Process in the background sending necessary data to the pormptworker function
    thr = Thread(target=prompt_worker, args=(response_url,text,user_name,channel,))
    thr.start()

    return response


#returning user information based on the user_email
@app.route('/user_info', methods=['POST'])
def user_info():
    data = request.form
    text = data.get('text')

    if ';' in text:
        return "Do not use ;"
  
    else:

        #execute the query based on the user_email inserted by the request
        query="SELECT * FROM test_1.Users WHERE Email=%s ORDER BY FirstName"
        cursor.execute(query,(text,))

        #formatting result
        result=cursor.fetchall()
        result_json=formatting(result=result,cursor=cursor)


    if len(result) >= 0:
        return jsonify({"blocks": result_json,"response_type": "ephemeral"})
    
    else :
         return "No user with that email in our database" 
#it will return stadium information based on different criterias 
@app.route('/stadium_info', methods=['POST'])
def stadium_info():

    data = request.form
    channel = data.get('channel_id')

    stadium = data.get('text')
    if ';' in stadium:
        return "Do not use ;"
    
    # Query to check by ID
    
    try:

        query = "SELECT * FROM test_1.stadiums WHERE id = %s"

        cursor.execute(query, (stadium,))
        result = cursor.fetchall()
        
        result_json=formatting(result=result,cursor=cursor)

        if len(result) > 0:
            return jsonify({"blocks": result_json,"response_type": "ephemeral"})
        
        #query to check based ob the stadium name(the first few letters)
        elif len(result) == 0:
            query2 = "SELECT * FROM test_1.stadiums WHERE stadiums.name LIKE %s"
            stadium += '%'
        
            cursor.execute(query2, (stadium,))
            result2 = cursor.fetchall()
        

            result_json2=formatting(result=result2,cursor=cursor)

            if len(result2) > 0 and len(result_json2) < 50:
                return jsonify({"blocks": result_json2,"response_type": "ephemeral"})

            elif len(result2) > 0 and len(result_json2) >= 50:

                client.files_upload_v2(token=os.getenv("SLACK_TOKEN"),content=f"{result_json2}",filename="results.txt",channel=channel)
                payload={"text":"Open this file to get the response","response_type":"ephemeral"}
                return payload  
        
    except Exception as e:
        return jsonify({"error": "Database operation failed", "message": str(e)})

    # If no results are found
    return "No data found"

#Gives courts information based on 2 criterias
@app.route('/court_info', methods=['POST'])
def court_info():
    data = request.form
    court = data.get('text')
    channel = data.get('channel_id')

    # Query to check by ID
    query = "SELECT id,name,status,defaultRate,openTime,closeTime,category,rateAutoUpdated,minPrice,maxPrice,onRequest,commission,commissionInPercent,sendDailySchedule,userId,createdById,userEmail,userLangKey,types,stadiumId,stadiumName,courtGrossRate,maxBookingLimit,allowMultiUnit,showMinMaxPlayers,showLongTermReservation,minReservationDurationMinutes,minReservationTimeIncreaseMinutes,destinationUrl,createdAt,updatedAt,lengthM,widthM FROM test_1.Court WHERE id = %s"
    if ';' in court:
         return "Do not use ;"
    
    try:
        cursor.execute(query, (court,))

        result = cursor.fetchall()
        result_json1=formatting(result=result,cursor=cursor)
  
        if len(result) > 0:
            return jsonify({"blocks": result_json1,"response_type": "ephemeral"})              
        # Query to check by name if no ID match is found
        query2 = "SELECT * FROM test_1.Court WHERE name LIKE %s"
        court += '%'

        cursor.execute(query2, (court,))
        result2 = cursor.fetchall()

        result_json2=formatting(result=result2,cursor=cursor)
  
        if len(result2) > 0 and len(result_json2) < 50:
                return jsonify({"blocks": result_json2,"response_type": "ephemeral"})

        elif len(result2) > 0 and len(result_json2) >= 50:

                client.files_upload_v2(token=os.getenv("SLACK_TOKEN"),content=f"{result_json2}",filename="results.txt",channel=channel)
                payload={"text":"Open this file to get the response","response_type":"ephemeral"}
                return payload


    except Exception as e:

        # Handling any exceptions that occur during database operations
        return jsonify({"error": "Database operation failed", "message": str(e)}), 500

    # If no results are found
    return "No data found"
#same as stadiuminfo and courtinfo
@app.route("/transactions",methods=['POST'])
def transaction_info():

    data = request.form
    transaction = data.get('text')

    if ';'  in transaction:
         return "Do not use ;"
    
    query="SELECT * FROM test_1.transactions Where Transaction_ID = %s"
    
    try:
        cursor.execute(query, (transaction,))
        result = cursor.fetchall()
        result_json=formatting(result=result,cursor=cursor)

        if len(result) > 0:
            return jsonify({"blocks": result_json,"response_type": "ephemeral"})  
        
        #select the last 5 transactions based on the startingdate  
        today = date.today()
        yesterday = today - timedelta(days = 1) 
        query2 = f"SELECT * FROM test_1.transactions WHERE StartDate= '{today}' OR StartDate ='{yesterday}' ORDER BY Start_Time DESC LIMIT 5 "

        
        cursor.execute(query2)
        result2 = cursor.fetchall()
        result_json2=formatting(result=result2,cursor=cursor)
 

        if len(result2) > 0:
            return jsonify({"blocks": result_json2,"response_type": "ephemeral"})     


    except Exception as e:
        return jsonify({"error": "Database operation failed", "message": str(e)})



'''////////////////////////////////////////////////////////'''

#general information of the company,could add some more commands if necessary
@app.route("/list",methods=['POST'])
def list():
     text='''list necessary information related to the chatbot
'''
     return text


@app.route("/info",methods=['POST'])
def info():
     
     text=f'''Insert information from the company'''
     return (text)
#start the application, need to configure it with server port and host(SERVER_NAME)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


