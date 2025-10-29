# Unity-Chatting-Agent
An implementation of an A2A chatting agent with access to MCP servers that you can talk to in Unity


## What is this repo?
For my Lehigh RiVR Project group, we were considering what features we could do with AI in our games. One of the things that we always wanted to do was to make a specialized chatbot that could answer player's specific questions that has to do with real time data and more specific information that you could only search online. 

So to test out, I decided to make a simple prototype of a chatbot agent with A2A and MCP protocols, specifically to the NWS and the usgs watershed mcp servers, both of which were related to our projects and didn't need any special accesses like API Keys and were free to use. Sadly though we decided not to use any AI features for now in our game due to the latency of the chatbot's answer, and the fact that it needs access to wifi to do so.


## What you need to run
For this project I am currently using Gemini to run and process all the logic, but you can certainly swap it out for a different model. If you do use Gemini though you will need an API key for it, as well as access to Google Cloud Project to run and test.

- **GOOGLE_API_KEY**: api key to use the Gemini model
- **GOOGLE_GENAI_MODEL**: the specific model of gemini to use, I used "gemini-2.0-flash-001"
- **GOOGLE_GENAI_USE_VERTEXAI**: to set using the ai model to true, set to "TRUE"
- **GOOGLE_CLOUD_PROJECT**: the name of your google cloud project
- **GOOGLE_CLOUD_LOCATION**: the location of your google cloud project

