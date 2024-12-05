import firebase_admin
from firebase_admin import credentials
from firebase_admin import  db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://automaticdooropenclose-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "321654":
        {
            "name": "Sampada Khopade",
            "major":"AI",
            "joining_year":2022,
            "total_in_time":6,
            "standing":"A",
            "year":4,
            "last_in_time":"2023-12-23 00:54:34"
        },

    "963852":
        {
            "name": "Elon Musk",
            "major":"Physics",
            "joining_year":1961,
            "total_in_time":10,
            "standing":"B",
            "year":4,
            "last_in_time":"2023-12-23 00:54:34"
        },
}
for key,value in data.items():
    ref.child(key).set(value)