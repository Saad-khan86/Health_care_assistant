from langchain_google_genai import ChatGoogleGenerativeAI
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow # type: ignore
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
import os
from typing import List
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# LLM
llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

# State
class State(MessagesState):
    summary: str

def schedule_appointment(
    summary: str,
    location: str,
    description: str,
    start_datetime: str,
    end_datetime: str,
    attendees: List[str],
) -> str:

    """set schedule_appointment"""

    scopes = ["https://www.googleapis.com/auth/calendar"]

    TOKEN_PATH = './token.json'
    credentials_file = './credentials.json'

    creds = None
    # Load or authenticate credentials
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, scopes
            )

            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

            auth_url, _ = flow.authorization_url(prompt='consent')
            print(f"Please visit this URL to authorize this application: {auth_url}")

            # Prompt the user to paste the authorization code
            authorization_code = input("Enter the authorization code: ")
            flow.fetch_token(code=authorization_code)
            creds = flow.credentials



        with open(TOKEN_PATH, 'w') as token_file:
            token_file.write(creds.to_json())

    try:
        service = build('calendar', 'v3', credentials=creds)
        # Define event details
        event = {
            'summary': summary,
            'location': location,
            'description': description,
            'start': {
                'dateTime': start_datetime,
                'timeZone': 'Asia/Karachi'
            },
            'end': {
                'dateTime': end_datetime,
                'timeZone': 'Asia/Karachi'
            },
            'attendees': [{'email': email} for email in attendees]
        }
        # Create event
        event = service.events().insert(calendarId='primary', body=event).execute()
        return f"Appointment set. plzz see Appointment schedule: {event.get('htmlLink')}"

    except HttpError as error:
        return f"An error occurred: {error}"


llm_with_tools = llm.bind_tools([schedule_appointment])

def analyze_symptoms(state: State):

    summary = state.get("summary", "")

    messages = []

    system_message = SystemMessage(content=(
        f"If the user asks about past conversations, Use the following:'Summary of earlier conversation: {summary}'"
        "You are a healthcare assistant focused solely on health-related queries, including medical conditions, treatments, wellness"
        "and fitness. For medical emergencies, advise contacting a professional. Analyze symptoms, predict diseases, and recommend treatments accurately"
        "Redirect non-health queries politely and respond user-friendly while maintaining professionalism"
        "Assist in scheduling appointments by gathering details like date, time, email, and formatting them in ISO format like: YYYY-MM-DDThh:mm:ss+00:00 don't tell to user about formating"
        "Follow a structured workflow to ensure clarity and professionalism while creating summaries, locations, and descriptions for schedule_appointment"
        "**set schedule_appointment**: "
        "- Once you have all the required information, call the tool `schedule_appointment` with the following parameters: "
        "- `summary`: get from user."
        "- `location`: get from user."
        "- `description`: Generated from the conversation context."
        "- `start_datetime`: The formatted start date and time."
        "- `end_datetime`: 1 hour after the start date and time."
        "- `attendees`: email addresses provided by the user."
        "Example Workflow: "
        "- User: 'I want to set an appointment.' "
        "- Assistant: 'Please provide the date and time for the appointment.' "
        "- User: '12 Dec 2024, 3.00 AM.' "
        "- Assistant: *Converts to ISO format:`2024-12-22T15:00:00+00:00` and calculates `end_datetime` as `2024-12-22T16:00:00+00:00`. "
        "- Assistant: 'Please provide the email address of the attendee(s).' "
        "- User: 'example@gmail.com.' "
        "- Assistant: *is this information correct ?*"
        "- User: 'yes'."
        "- Assistant: *Calls the tool `schedule_appointment` with the gathered information.*"))

    messages.insert(0, system_message)

    messages.extend(state["messages"])

    response = llm_with_tools.invoke(messages)

    return {"messages": response}

def create_summary(state: State):

  summary = state.get("summary", "")

  if summary:
    summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-6]]

  else:
      summary_message = "Create a summary of the conversation above:"
      messages = state["messages"] + [HumanMessage(content=summary_message)]
      response = llm.invoke(messages)
      delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-6]]

  return {"summary": response.content, "messages": delete_messages}


builder: StateGraph = StateGraph(State)

builder.add_node("analyze_symptoms", analyze_symptoms)
builder.add_node("create_summary", create_summary)
builder.add_node("tools", ToolNode(tools=[schedule_appointment]))

builder.add_edge(START, "analyze_symptoms")
builder.add_conditional_edges(
    "analyze_symptoms",
    tools_condition,  # Condition for tool usage
    {"tools": "tools", END: END}  # Define the path map
)
builder.add_edge("tools", END)
builder.add_edge("analyze_symptoms", "create_summary")
builder.add_edge("create_summary", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])
display(Image(graph.get_graph().draw_mermaid_png()))