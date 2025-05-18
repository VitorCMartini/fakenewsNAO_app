import os
import streamlit as st
from google.generativeai import GenerativeModel
from datetime import date
import warnings
import json

warnings.filterwarnings("ignore")

# Defina sua chave de API Gemini aqui
api_key = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key

MODEL_ID = "gemini-2.0-flash"

# Funções para memória do chat
MEMORIA_PATH = "chat_memoria.json"

def carregar_memoria():
    if os.path.exists(MEMORIA_PATH):
        with open(MEMORIA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def salvar_memoria(memoria):
    with open(MEMORIA_PATH, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=2)

# Função para enviar prompt ao Gemini
def gerar_resposta(prompt):
    model = GenerativeModel(MODEL_ID)
    response = model.generate_content(prompt)
    return response.text

##########################################
# --- Agente 1: Buscador de Notícias --- #
##########################################
def agente_buscador(topico, data_de_hoje):
    prompt = f"""
Você é um assistente de pesquisa especializado em determinar a probabilidade de veracidade de um tópico com base na análise de notícias recentes e históricas.

Sua tarefa é buscar notícias relacionadas ao seguinte tópico:

Tópico: {topico}
Data de hoje: {data_de_hoje}

Ao realizar a busca e analisar os resultados, siga estas etapas:

1. Busque as últimas notícias relevantes sobre o tópico: Concentre-se em notícias publicadas recentemente (idealmente nos últimos meses e de fontes seguras e/ou institucionais).
2. Analise a probabilidade de veracidade com base na comparação:
   - Concordância: Verifique se múltiplas fontes confiáveis reportam informações semelhantes sobre o tópico. Quanto maior a concordância entre fontes reputáveis, maior a probabilidade de veracidade.
   - Tom: Avalie o tom das notícias. Notícias com tom neutro e factual de fontes confiáveis tendem a indicar maior probabilidade de veracidade do que notícias com tom sensacionalista ou excessivamente opinativo.
   - Entusiasmo/Cobertura: Se um tópico gerar uma grande quantidade de notícias e discussões de fontes diversas e confiáveis, isso pode indicar sua relevância e potencial veracidade (embora não garanta).
3. Lide com poucos resultados ou falta de entusiasmo:
   - Se houver poucas notícias recentes ou pouco entusiasmo em torno do tópico, realize uma busca por notícias que possam ter desencadeado o tema, buscando informações de até dois anos antes.
   - Avalie os fatos apresentados nessas notícias mais antigas para entender a origem e o desenvolvimento do tema.
4. Formate cada notícia encontrada: Para cada notícia relevante, inclua no seu relatório:
   - Data de Publicação: (Extraída da notícia)
   - Autor/Fonte: (Nome do veículo ou autor, se disponível)
   - Título da Notícia:
   - Breve Resumo/Trecho Relevante: (Um ou dois parágrafos destacando as informações chave)

Seu relatório deve ser um resumo das notícias encontradas e sua avaliação inicial da probabilidade de veracidade do tópico com base na sua análise comparativa, tom e cobertura.
Mencione se houve poucos resultados recentes e como você procedeu para investigar a origem do tema.
"""
    return gerar_resposta(prompt)

################################################
# --- Agente 2: Avaliador de notícias --- #
################################################
def agente_avaliador(topico, noticias_buscadas):
    prompt = f"""
Você é um assistente de análise de informações, responsável por avaliar o relatório produzido pelo 'agente_buscador' sobre a veracidade de um tópico.

Tópico: {topico}

Relatório do agente_buscador:
{noticias_buscadas}

Com base no relatório acima, realize as seguintes ações:

1. Classificação Hierárquica da Veracidade: Classifique a probabilidade de veracidade da informação em uma escala (por exemplo: Alta Probabilidade, Média Probabilidade, Baixa Probabilidade, Improvável, Não Conclusivo). Justifique sua classificação com base na concordância, tom e entusiasmo observados nas notícias.
2. Análise Qualitativa da Veracidade: Descreva qualitativamente o nível de suporte factual encontrado. Existem evidências concretas e verificáveis? As fontes são confiáveis e consistentes? Há contradições ou informações conflitantes?
3. Análise Quantitativa da Veracidade: Se possível, quantifique a força do suporte (por exemplo: "8 em cada 10 fontes principais reportam informações consistentes").
4. Identificação de Viés Ideológico: Analise as fontes das notícias em busca de possíveis vieses ideológicos que possam influenciar a forma como a informação é apresentada. Mencione os possíveis vieses identificados e como eles podem afetar a interpretação da informação.
5. Contextualização da Informação: Com base nas notícias e em seu conhecimento geral, forneça o contexto provável em que a informação está inserida. Quais eventos ou situações podem estar relacionados a essa informação?
6. Conteúdo de Apoio para o Agente Redator: Forneça uma lista concisa de pontos chave, fatos confirmados (se houver), evidências claras e o contexto identificado que o 'agente_redator' pode usar para escrever seu relatório. Priorize informações com forte suporte factual.

Seu relatório deve ser uma análise estruturada da probabilidade de veracidade, viés ideológico, contexto e um resumo de conteúdo chave para o 'agente_redator'.
"""
    return gerar_resposta(prompt)

######################################
# --- Agente 3: Redator do Post --- #
######################################
def agente_redator(topico, avaliacao_critica):
    prompt = f"""
Você é um redator de relatórios informativos, responsável por apresentar um parecer claro e direto sobre a veracidade de uma informação com base na análise fornecida pelo 'agente_avaliador'.

Tópico: {topico}

Relatório do agente_avaliador:
{avaliacao_critica}

Com base no relatório acima, redija um relatório que inclua os seguintes elementos:

1. Parecer sobre a Veracidade: Apresente de forma clara o parecer sobre a probabilidade de veracidade da informação (usando a classificação do 'agente_avaliador').
2. Veículo de Disseminação (se conhecido): Mencione qual foi o veículo de informação que disseminou a notícia original (se essa informação estiver disponível no input ou nos relatórios anteriores).
3. Contexto da Informação: Apresente o contexto em que a informação provavelmente está inserida, conforme identificado pelo 'agente_avaliador'.
4. Informações Esclarecedoras: Forneça informações chave que ajudem o leitor a entender a mensagem com maior facilidade. Baseie-se nos fatos confirmados e nas evidências claras identificadas pelo 'agente_avaliador'.
5. Fontes Principais (opcional): Se relevante e conciso, você pode mencionar as principais fontes que sustentam o parecer (sem entrar em detalhes exaustivos).

Seu relatório deve ser um texto claro, direto e informativo, resumindo a análise de veracidade, o contexto e as informações essenciais para o entendimento do leitor. Evite jargões e seja objetivo.
"""
    return gerar_resposta(prompt)

##########################################
# --- Agente 4: Revisor de Qualidade --- #
##########################################
def agente_revisor(topico, rascunho_gerado):
    prompt = f"""
Você é um revisor de textos com foco em clareza e precisão. Sua tarefa é revisar o relatório escrito pelo 'agente_redator' em busca de frases ambíguas que possam levar a interpretações equivocadas.

Tópico: {topico}

Rascunho do agente_redator:
{rascunho_gerado}

Ao revisar o relatório, siga estas etapas:

1. Leia atentamente o relatório.
2. Identifique frases que possam ter múltiplos significados ou que não sejam claras o suficiente. Procure por pronomes com referentes obscuros, construções gramaticais complexas que dificultam a compreensão, termos vagos ou imprecisos, e qualquer outra formulação que possa gerar confusão no leitor.
3. Sugira revisões para tornar as frases mais claras e diretas. Seu objetivo é eliminar qualquer ambiguidade e garantir que a mensagem seja transmitida de forma inequívoca.
4. Mantenha o tom e o estilo do relatório original o máximo possível, focando apenas na eliminação da ambiguidade.

Seu relatório deve ser uma lista das frases ambíguas encontradas e suas sugestões de revisão para maior clareza. Se não encontrar ambiguidades, informe isso explicitamente.
"""
    return gerar_resposta(prompt)

# --- Interface Streamlit com memória ---
st.set_page_config(page_title="Verificador de Fake News", layout="wide")
st.title("🚀 Sistema de Verificação de Notícias com Gemini")

# Carregar histórico do chat
memoria = carregar_memoria()

topico = st.text_input("Digite o TÓPICO sobre o qual você quer verificar a informação:")

if st.button("Verificar notícia"):
    if not topico:
        st.warning("Você esqueceu de digitar o tópico!")
    else:
        data_de_hoje = date.today().strftime("%d/%m/%Y")
        st.info("Buscando notícias...")
        noticias_buscadas = agente_buscador(topico, data_de_hoje)
        st.success("Notícias buscadas!")
        st.info("Avaliando notícias...")
        avaliacao_critica = agente_avaliador(topico, noticias_buscadas)
        st.success("Avaliação crítica pronta!")
        st.info("Redigindo relatório...")
        redacao = agente_redator(topico, avaliacao_critica)
        st.markdown("### Rascunho Gerado")
        st.markdown(redacao)
        st.info("Revisando relatório...")
        redacao_revisada = agente_revisor(topico, redacao)
        st.markdown("### Rascunho Revisado")
        st.markdown(redacao_revisada)

        # Salvar no histórico/memória
        memoria.append({
            "data": data_de_hoje,
            "topico": topico,
            "noticias_buscadas": noticias_buscadas,
            "avaliacao_critica": avaliacao_critica,
            "redacao": redacao,
            "redacao_revisada": redacao_revisada
        })
        salvar_memoria(memoria)

# Exibir histórico do chat
if memoria:
    st.markdown("---")
    st.markdown("## Histórico de Consultas")
    for i, item in enumerate(reversed(memoria), 1):
        with st.expander(f"{item['data']} - {item['topico']}"):
            st.markdown("**Rascunho Revisado:**")
            st.markdown(item["redacao_revisada"])