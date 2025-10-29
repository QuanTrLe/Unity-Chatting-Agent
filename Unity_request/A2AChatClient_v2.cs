using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System.Collections;
using System.Text;
using TMPro; // Use TextMeshPro

public class A2AChatClient_v2 : MonoBehaviour
{
    // --- Configuration ---
    [Tooltip("The full URL of the agent's chat endpoint (e.g., http://localhost:10004/simple_chat)")]
    [SerializeField] private string agentUrl = "http://localhost:10004/simple_chat";
    [SerializeField] private int requestTimeout = 15; // 15 seconds

    // --- UI References ---
    public TMP_InputField queryInputField;
    public Button sendButton;
    public TMP_Text chatDisplay;

    public ScrollRect chatScrollRect;

    // --- State ---
    private string sessionId;
    private bool isWaitingForResponse = false;

    // --- C# Classes to match the JSON ---
    [System.Serializable]
    private class ChatRequest
    {
        public string query;
        public string session_id;
    }

    [System.Serializable]
    private class ChatResponse
    {
        public string response;
        public string error;
    }

    void Start()
    {
        sessionId = System.Guid.NewGuid().ToString();

        sendButton.onClick.AddListener(OnSendClicked);
        queryInputField.onSubmit.AddListener((_) => OnSendClicked());

        AddToChat("Unity Client Initialized. Session: " + sessionId, Color.green);
    }

    private void OnSendClicked()
    {
        // Don't send if we're already waiting
        if (isWaitingForResponse)
        {
            return;
        }

        string query = queryInputField.text;
        if (string.IsNullOrWhiteSpace(query))
        {
            return;
        }

        AddToChat("You: " + query, Color.white);
        queryInputField.text = "";

        // Start the web request
        StartCoroutine(SendChatRequestCoroutine(query));
    }

    IEnumerator SendChatRequestCoroutine(string userQuery)
    {
        // 1. Set "loading" state
        SetLoadingState(true);

        // 2. Create the request payload
        ChatRequest payload = new ChatRequest
        {
            query = userQuery,
            session_id = sessionId
        };
        string jsonPayload = JsonUtility.ToJson(payload);
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);

        // 3. Create the UnityWebRequest
        UnityWebRequest request = new UnityWebRequest(agentUrl, "POST");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");
        request.timeout = requestTimeout;

        // --- NEW: Better Debugging ---
        Debug.Log($"[A2AChatClient] Sending to {agentUrl}: {jsonPayload}");

        // 4. Send the request
        yield return request.SendWebRequest();

        // 5. Handle the response
        try
        {
            if (request.result == UnityWebRequest.Result.ConnectionError)
            {
                Debug.LogError($"[A2AChatClient] Connection Error: {request.error}");
                AddToChat($"Error (Connection): {request.error}", Color.red);
            }
            else if (request.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.LogError($"[A2AChatClient] Protocol Error: {request.error}");
                AddToChat($"Error (Server): {request.error}", Color.red);
            }
            else if (request.result == UnityWebRequest.Result.Success)
            {
                string jsonResponse = request.downloadHandler.text;
                Debug.Log($"[A2AChatClient] Received: {jsonResponse}");

                ChatResponse response = JsonUtility.FromJson<ChatResponse>(jsonResponse);

                if (!string.IsNullOrEmpty(response.response))
                {
                    AddToChat("Agent: " + response.response, Color.cyan);
                }
                else if (!string.IsNullOrEmpty(response.error))
                {
                    Debug.LogWarning($"[A2AChatClient] Server returned an error message: {response.error}");
                    AddToChat("Server Error: " + response.error, Color.red);
                }
                else
                {
                    Debug.LogError($"[A2AChatClient] Received empty or invalid JSON response: {jsonResponse}");
                    AddToChat("Error: Received invalid response from server.", Color.red);
                }
            }
        }
        finally
        {
            // 6. Unset "loading" state
            SetLoadingState(false);
        }
    }

    // Helper to add text to the chat display
    void AddToChat(string text, Color color)
    {
        chatDisplay.text += $"<color=#{ColorUtility.ToHtmlStringRGB(color)}>{text}</color>\n\n";
        StartCoroutine(ForceScrollToBottom());
    }

    IEnumerator ForceScrollToBottom()
    {
        // Wait for the UI system to update the layout (after text is added)
        yield return new WaitForEndOfFrame();

        // Force the scroll position to the bottom (0 = bottom, 1 = top)
        if (chatScrollRect != null)
        {
            chatScrollRect.verticalNormalizedPosition = 0f;
        }
    }

    // Helper to manage UI state
    void SetLoadingState(bool isLoading)
    {
        isWaitingForResponse = isLoading;
        sendButton.interactable = !isLoading;
        queryInputField.interactable = !isLoading;

        if (!isLoading)
        {
            // Re-focus the input field so the user can type again
            queryInputField.Select();
            queryInputField.ActivateInputField();
        }
    }
}