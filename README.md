## Deep Fashion API  

Implementation for a web server that accepts an image and returns product listings with similar attributes.  

![Architecture Diagram](/etc/architecture.png?raw=true)  

The issues system is being used as a to do list. Claim an issue to take ownership, and resolve an issue to mark complete.  

Local Setup:  
1. Install Python 3.x and MongoDB.  
2. Run mongod.exe to start a local mongo server.  
3. Run main.py to start the local python server.  
4. Send a POST with an image to 127.0.0.1:80 (use Postman or script).  

To send a request to the production server, send a POST with an image to 138.197.9.186:80. The response is JSON formatted product listings.
