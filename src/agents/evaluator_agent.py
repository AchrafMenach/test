from crewai import Agent
from langchain_groq import ChatGroq
import os

class EvaluatorAgent:
    def __init__(self, llm: ChatGroq):
        self.llm = llm

    def create_agent(self) -> Agent:
        return Agent(
            role="Évaluateur Expert",
            name="AnswerEvaluator",
            goal="""Fournir des évaluations précises et pédagogiques des réponses mathématiques.
                Identifier clairement les erreurs et fournir des explications détaillées.""",
            backstory="""Professeur agrégé de mathématiques avec 15 ans d\"expérience
                dans l\"enseignement secondaire et supérieur. Spécialiste de la pédagogie différenciée.""",
            llm=self.llm,
            verbose=False,
            max_iter=15,
            memory=True
        )


