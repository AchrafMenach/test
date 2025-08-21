from crewai import Agent
from langchain_groq import ChatGroq
import os

class PersonalCoachAgent:
    def __init__(self, llm: ChatGroq):
        self.llm = llm

    def create_agent(self) -> Agent:
        return Agent(
            role="Coach Personnel en Mathématiques",
            name="PersonalMathCoach",
            goal="""Fournir un accompagnement personnalisé, des encouragements 
            et des stratégies d\"apprentissage adaptées à chaque étudiant""",
            backstory="""Ancien professeur de mathématiques devenu coach scolaire,
            spécialisé dans la motivation et la résolution des blocages psychologiques
            liés à l\"apprentissage des mathématiques. Utilise des techniques de
            pédagogie positive et de renforcement des compétences.""",
            llm=self.llm,
            verbose=False,
            memory=True,
            max_iter=10
        )


