from flask import Flask, render_template, flash, redirect, url_for, request
from py2neo import Graph, DatabaseError
import aiml
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import requests
from bs4 import BeautifulSoup


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace 'your_secret_key' with your actual secret key

try:
    graph = Graph("bolt://localhost:7690", auth=("neo4j", "12345678"))  # Update with your Neo4j connection details
except DatabaseError as e:
    print(f"Error connecting to Neo4j database: {e}")
    exit(1)

Kernel = aiml.Kernel()
for filename in os.listdir("aiml_files"):
    if filename.endswith(".aiml"):
        Kernel.learn("aiml_files/" + filename)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

model = joblib.load(os.path.abspath("ML/gender_detect.pkl"))
vectorizer = CountVectorizer()

def predict_gender(name):
    vocabulary = joblib.load(os.path.abspath("ML/vocabulary.pkl"))
    vectorizer.vocabulary_ = vocabulary

    name_vectorized = vectorizer.transform([name])
    predicted_gender = model.predict(name_vectorized)[0]
    return predicted_gender

def preprocess_input(input_text):
    tokens = nltk.word_tokenize(input_text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    processed_input = ' '.join(lemmatized_tokens)
    return processed_input

@app.route("/index1")
def index1():
    return render_template("index1.html")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]

        # Validate input
        if not uname or not passw:
            flash("Please enter both username and password.", "danger")
            return redirect(url_for("login"))

        try:
            # Example Neo4j query to retrieve user
            query = "MATCH (u:User {username: $username, password: $password}) RETURN u"
            result = graph.run(query, username=uname, password=passw).data()

            if result:
                return redirect(url_for("index1"))
            else:
                flash("Invalid username or password. Please try again.", "danger")
                return redirect(url_for("login"))

        except DatabaseError as e:
            flash(f"Error accessing user data: {e}", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fname = request.form['fname']
        lname = request.form['lname']
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']
        gender = predict_gender(fname)

        try:
            query = """
                CREATE (u:User {firstname: $firstname, lastname: $lastname, gender: $gender,
                username: $username, email: $email, password: $password})
            """
            graph.run(query, firstname=fname, lastname=lname, gender=gender,
                      username=uname, email=mail, password=passw)

            # Create two Person nodes and a relationship between them
            query_relationship = """
                CREATE (p1:Person {name: $name1}), (p2:Person {name: $name2})
                CREATE (p1)-[:FRIEND]->(p2)
            """
            graph.run(query_relationship, name1=fname, name2=lname)

            query_father_relationship = """
                MATCH (child:Person {name: $childName}), (father:Person {name: $fatherName})
                CREATE (child)-[:FATHER]->(father)
            """
            graph.run(query_father_relationship, childName=fname, fatherName=lname)

            query_mother_relationship = """
                MATCH (child:Person {name: $childName}), (mother:Person {name: $motherName})
                CREATE (child)-[:MOTHER]->(mother)
            """
            graph.run(query_mother_relationship, childName=fname, motherName=lname)

            query_sister_relationship = """
                MATCH (p1:Person {name: $name1}), (p2:Person {name: $name2})
                CREATE (p1)-[:SISTER]->(p2)
            """
            graph.run(query_sister_relationship, name1=fname, name2=lname)

            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))

        except DatabaseError as e:
            flash(f"Error creating user: {e}", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")

@app.route("/api", methods=["POST"])
def api():
    user_input = request.form.get("msg")
    bot_response = Kernel.respond(user_input)

    try:
        # Example Neo4j query
        result = graph.run("MATCH (n:Node) RETURN n").data()
        # Process the result and perform any necessary operations

    except DatabaseError as e:
        flash(f"Error accessing data: {e}", "danger")
        return redirect(url_for("chatbot"))

    return bot_response

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "POST":
        user_input = request.form.get("user_input")

        # Preprocess user input
        processed_input = preprocess_input(user_input)

        # Pass the processed input to the AIML Kernel for a response
        bot_response = Kernel.respond(processed_input)

        try:
            # Example Neo4j query
            result = graph.run("MATCH (n:Node) RETURN n").data()
            # Process the result and perform any necessary operations

        except DatabaseError as e:
            flash(f"Error accessing data: {e}", "danger")
            return redirect(url_for("chatbot"))

        return render_template("chatbot.html", user_input=user_input, bot_response=bot_response)

    return render_template("chatbot.html")

def set_predicate(predicate, value):
    # Perform any necessary operations to set the predicate value
    # For example, you can store the predicate-value pair in a dictionary or database
    # Here, we'll simply print the predicate and value
    print(f"Set {predicate} to '{value}'")
    # Update the predicate value to "soshi chatbot"
    value = "soshi chatbot"
    print(f"Updated {predicate} to '{value}'")



# Rest of your code...

def get_predicate(predicate, user):
    # Retrieve the predicate value for the user from a data source
    # Here, we'll use a dictionary to store the predicate-value pairs for each user
    user_predicates = {
        "user1": {
            "predicate1": "Value for predicate1",
            "predicate2": "Value for predicate2"
        },
        "user2": {
            "predicate1": "Value for predicate1",
            "predicate2": "Value for predicate2"
        },
        # Add more users and their corresponding predicate values as needed
    }

    # Check if the user exists in the dictionary
    if user in user_predicates:
        user_values = user_predicates[user]
        # Check if the predicate exists for the user
        if predicate in user_values:
            return user_values[predicate]

    # Return a default value if the user or predicate is not registered
    return "Unknown"

# Rest of your code...



def scrape_wikipedia(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Find the relevant elements in the HTML using BeautifulSoup methods
    # Extract the desired information from the elements
    # Return the extracted data as needed

    # Example: Extract the page title
    title = soup.find("h1", id="firstHeading").text
    return title

@app.route("/scrape", methods=["GET", "POST"])
def scrape():
    if request.method == "POST":
        url = request.form["url"]
        set_predicate_value = request.form["set_predicate"]
        get_predicate_value = request.form["get_predicate"]

        if not url or not set_predicate_value or not get_predicate_value:
            flash("Please enter all the required information.", "danger")
            return redirect(url_for("scrape"))

        try:
            # Scrape Wikipedia page
            scraped_data = scrape_wikipedia(url)
            # Set the predicate value
            set_predicate(set_predicate_value, scraped_data)
            # Get the predicate value
            retrieved_data = get_predicate(get_predicate_value)

            return render_template("scrape.html", scraped_data=scraped_data, retrieved_data=retrieved_data)

        except requests.RequestException as e:
            flash(f"Error scraping data: {e}", "danger")
            return redirect(url_for("scrape"))

    return render_template("scrape.html")

def find_definition_word(query):
    if query.startswith("what is the definition of "):
        return query[26:]
    if query.startswith("define "):
        return query[7:]
    if query.startswith("tell me about "):
        return query[13:]
    return ""

def get_sentiment(text, sid=None):
    sentiment_scores = sid.polarity_scores(text)
    compound_score = sentiment_scores["compound"]
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=9000, debug=True)



