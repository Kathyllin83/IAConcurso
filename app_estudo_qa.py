import streamlit as st
import json
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random 


# --- Configurações Globais ---
DATA_FILE = 'banco_de_perguntas.json'
LIMIAR_SIMILARIDADE = 0.25 
IMAGEM_LARGURA_PADRAO = 200 # Ajustado para um tamanho mais visível para imagens
NUM_ALTERNATIVAS = 4 # Total de alternativas por questão (1 correta + 3 incorretas)


# --- Caminho Local do Logo ---
LOCAL_LOGO_PATH = "logo.png" 


# --- Definição das Cores Verdes Escuras (Valores Hexadecimais) ---
COR_FUNDO_PRINCIPAL = "#1A4314"       # Verde escuro profundo
COR_TEXTO_PRINCIPAL = "#FFFFFF"       # Branco
COR_FUNDO_SECUNDARIO = "#2B5C22"      # Verde escuro intermediário (para sidebar, inputs)
COR_PRIMARIA_ELEMENTOS = "#3D6635"    # Verde um pouco mais claro (para botões, links)
COR_TEXTO_SECUNDARIO = "#CCCCCC"      # Cinza claro para texto secundário

# --- Funções de Backend ---

def carregar_perguntas():
    """Carrega as perguntas e respostas de um arquivo JSON."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning(f"Ops! 🧐 Erro ao ler seu arquivo de conhecimento ({DATA_FILE}). Parece que está vazio ou com um formato estranho. Não se preocupe, vamos começar do zero! ✨")
            return []
    return []

def salvar_pergunta(pergunta, resposta, tags=None, imagem_url=None):
    """Adiciona uma nova pergunta e resposta ao banco de dados e salva."""
    banco_dados = carregar_perguntas()
    novo_item = {
        "pergunta": pergunta,
        "resposta": resposta,
        "tags": tags if tags is not None else [],
        "imagem_url": imagem_url if imagem_url else "" # Adicionado campo para imagem
    }
    banco_dados.append(novo_item)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(banco_dados, f, indent=4, ensure_ascii=False)
    st.balloons() # Balões para celebrar o salvamento!
    st.success("🎉 Flashcard salvo com sucesso no seu cérebro digital! 🎉")


def buscar_pergunta_existente(termo_busca):
    """
    Busca no banco de dados por perguntas ou respostas que contenham o termo.
    Retorna uma lista de itens que correspondem.
    """
    termo_busca_lower = termo_busca.lower()
    resultados = []
    banco_dados = carregar_perguntas() # Recarrega para ter certeza de que está atualizado
    for item in banco_dados:
        # Verifica se o termo está na pergunta, resposta ou nas tags
        if termo_busca_lower in item['pergunta'].lower() or \
           termo_busca_lower in item['resposta'].lower() or \
           any(termo_busca_lower in tag.lower() for tag in item.get('tags', [])):
            resultados.append(item)
    return resultados


# --- Lógica de QA por Similaridade (TF-IDF e Cosseno) ---

@st.cache_data
def preencher_e_vetorizar_banco():
    """
    Carrega o banco de dados e vetoriza as perguntas para busca por similaridade.
    Cacheada para performance.
    """
    banco_dados = carregar_perguntas()
    
    # Assegura que o recurso 'stopwords' do NLTK esteja baixado
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        st.info("Parece que as 'stopwords' do NLTK (para português) ainda não estão por aqui... Baixando para você! 🚀")
        nltk.download('stopwords')

    textos_para_vetorizar = [item['pergunta'] for item in banco_dados]

    if not textos_para_vetorizar:
        # Retorna um vetorizador e matriz vazios se não houver dados
        return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('portuguese')), np.array([]), []

    vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('portuguese'), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(textos_para_vetorizar)
    
    return vectorizer, tfidf_matrix, banco_dados

# Carrega e vetoriza o banco de dados na inicialização do Streamlit
vectorizer, tfidf_matrix, banco_dados_qa = preencher_e_vetorizar_banco()


def responder_pergunta_qa(pergunta_usuario):
    """
    Busca a resposta mais relevante no banco de dados usando similaridade TF-IDF e Cosseno.
    """
    # Atualiza o banco de dados e a matriz TF-IDF (para refletir novas adições)
    # Refaz a vetorização para incorporar novas perguntas, se houver
    global vectorizer, tfidf_matrix, banco_dados_qa
    vectorizer, tfidf_matrix, banco_dados_qa = preencher_e_vetorizar_banco()

    if not banco_dados_qa or tfidf_matrix.size == 0:
        return "😔 Ops! Meu banco de conhecimentos está vazio. Que tal adicionar alguns flashcards primeiro? Assim posso aprender com você! 💡", None, None

    try:
        query_vector = vectorizer.transform([pergunta_usuario])
    except ValueError:
        return "🤔 Não consegui entender sua pergunta! Tente reformulá-la de um jeito diferente, por favor. 🙏", None, None

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    melhor_indice = similarity_scores.argmax()
    melhor_score = similarity_scores[melhor_indice]

    if melhor_score > LIMIAR_SIMILARIDADE:
        resposta_encontrada = banco_dados_qa[melhor_indice]['resposta']
        pergunta_original = banco_dados_qa[melhor_indice]['pergunta']
        imagem_url_encontrada = banco_dados_qa[melhor_indice].get('imagem_url', '')
        return resposta_encontrada, pergunta_original, imagem_url_encontrada
    else:
        return "🤷‍♀️ Desculpe, não encontrei uma resposta super relevante em meu conhecimento para sua pergunta. Que tal me ensinar algo novo? ✨", None, None


# --- Funções de Geração de Flashcards BASEADA EM REGRAS (NLTK) ---
def gerar_flashcard_simples_nltk(texto):
    """
    Gera um flashcard básico a partir de um texto usando NLTK (regras).
    """
    # Assegura que os recursos 'punkt' e 'averaged_perceptron_tagger' estejam baixados
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Falta o 'punkt' do NLTK! Baixando... 🛠️")
        nltk.download('punkt')
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        st.info("Falta o 'averaged_perceptron_tagger' do NLTK! Baixando... 🛠️")
        nltk.download('averaged_perceptron_tagger')
        
    try:
        sentencas = sent_tokenize(texto)
        if not sentencas:
            return None, None

        primeira_sentenca = sentencas[0]
        palavras = word_tokenize(primeira_sentenca)
        tags = pos_tag(palavras)

        substantivos = [word for word, tag in tags if tag.startswith('NN')] 
        
        pergunta = ""
        resposta = primeira_sentenca

        if substantivos:
            pergunta = f"O que é {substantivos[0]}?"
            if "é" in palavras or "são" in palavras:
                pergunta = f"O que é que está relacionado a '{substantivos[0]}' no texto?"
            else:
                pergunta = f"Qual é a informação principal sobre '{substantivos[0]}'?"
        else:
            pergunta = f"Qual é a ideia principal de: '{primeira_sentenca}'?"

        return pergunta, resposta
    except LookupError as e:
        st.error(f"⚠️ Erro NLTK: Um recurso necessário não foi encontrado. "
                 "Por favor, execute no seu terminal (com o ambiente virtual ativado): "
                 "`python -c \"import nltk; nltk.download('all')\"` para baixar tudo. "
                 f"Erro original: {e}")
        return None, None
    except Exception as e:
        st.error(f"🚫 Ocorreu um erro inesperado ao gerar o flashcard com NLTK: {e}")
        return None, None


# --- Interface do Streamlit ---

# Configurações da página - base theme 'dark' é bom para começar com cores escuras
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Cérebro de Pão", page_icon="🍞")



st.title("🧠 Cérebro de Pão! 🍞")
st.markdown("Bem-vindo(a) à sua ferramenta de estudo superpotente! 🚀")

# --- Adiciona o Logo Local na Sidebar ---
if os.path.exists(LOCAL_LOGO_PATH):
    st.sidebar.image(LOCAL_LOGO_PATH, width=150)
else:
    st.sidebar.warning("Logo não encontrado! Verifique o caminho: logo.png")

st.sidebar.markdown("---") 

st.sidebar.header("Escolha sua Aventura! 🗺️")
opcao_selecionada = st.sidebar.radio(
    "O que vamos aprender hoje?",
    ("🗣️ Fazer uma Pergunta à IA", "📝 Gerar Flashcard (IA Básica)", "📚 Consultar Flashcards", "➕ Adicionar Flashcard Manual", "❓ Modo Quiz (Múltipla Escolha)")
)

# Opção: Fazer uma Pergunta à IA
if opcao_selecionada == "🗣️ Fazer uma Pergunta à IA":
    st.header("Hora de Perguntar ao Gênio! 🧞")
    st.info("Curioso(a)? Digite sua pergunta e veja se a IA tem a resposta no seu banco de dados! 🤓")
    
    pergunta_do_usuario = st.text_input("Qual a sua dúvida hoje?", placeholder="Ex: Qual a capital da França?")
    
    if st.button("Obter Resposta! 💡"):
        if pergunta_do_usuario:
            with st.spinner("Pensando... pensando... 🧠"):
                resposta_qa, pergunta_encontrada_no_banco, imagem_url_qa = responder_pergunta_qa(pergunta_do_usuario)
            
            if pergunta_encontrada_no_banco:
                st.subheader("Resposta Mágica do Cérebro de Pão! ✨")
                st.success(f"{resposta_qa}")
                st.markdown(f"*Psst! Essa resposta veio da pergunta: \"{pergunta_encontrada_no_banco}\" que está no seu baú do conhecimento! 😉*")
                if imagem_url_qa:
                    st.image(imagem_url_qa, caption="Imagem do flashcard", width=IMAGEM_LARGURA_PADRAO)
            else:
                st.warning(f"{resposta_qa}")

        else:
            st.warning("Ops! Você esqueceu a pergunta! Por favor, digite algo. 😅")

# Opção: Gerar Flashcard (NLTK Básico)
elif opcao_selecionada == "📝 Gerar Flashcard (IA Básica)":
    st.header("Transformando Texto em Flashcards! 🚀")
    st.info("Cole um parágrafo aqui e deixe a IA criar um flashcard para você! Simples assim! 👇")
    texto_input = st.text_area("Seu texto mágico aqui:", height=150, placeholder="Cole um parágrafo sobre fotossíntese, por exemplo!")
    
    imagem_url_gerado = st.text_input("URL da Imagem (opcional):", placeholder="https://exemplo.com/imagem.png")

    if st.button("Criar Flashcard AGORA! 💫"):
        if texto_input:
            pergunta, resposta = gerar_flashcard_simples_nltk(texto_input)
            if pergunta and resposta:
                st.subheader("Seu Novo Flashcard Prontinho! 🎉")
                st.info(f"**Pergunta:** {pergunta}")
                st.success(f"**Resposta:** {resposta}")
                if imagem_url_gerado:
                    st.image(imagem_url_gerado, caption="Imagem do flashcard", width=IMAGEM_LARGURA_PADRAO)
                
                if st.button("Guardar este Tesouro (Flashcard)! 💾"):
                    salvar_pergunta(pergunta, resposta, ["gerado_por_nltk", "ia_basica"], imagem_url_gerado)
            else:
                st.warning("Ah, não! 😞 Não consegui criar um flashcard para este texto. Tente outro, por favor!")
        else:
            st.warning("Ei! 😮 Cadê o texto? Cole algo para eu trabalhar! 😉")


# Opção: Consultar Flashcards Existentes
elif opcao_selecionada == "📚 Consultar Flashcards":
    st.header("Revise seus Tesouros! 📖")
    st.info("Busque por qualquer palavra ou termo em seus flashcards já salvos. É como ter um mapa do conhecimento! 🗺️")
    termo_busca = st.text_input("O que você quer encontrar no seu baú do conhecimento?", placeholder="Ex: Brasil, fotossíntese, Roma")
    
    if st.button("Buscar no Baú! 🔎"):
        if termo_busca:
            resultados = buscar_pergunta_existente(termo_busca)
            if resultados:
                st.subheader(f"Encontrei {len(resultados)} tesouros para '{termo_busca}':")
                for i, item in enumerate(resultados):
                    st.markdown(f"### **Flashcard {i+1}:**")
                    st.markdown(f"- **Pergunta:** {item['pergunta']}")
                    st.markdown(f"- **Resposta:** {item['resposta']}")
                    if item.get('imagem_url'):
                        st.image(item['imagem_url'], caption=f"Imagem para: {item['pergunta']}", width=IMAGEM_LARGURA_PADRAO)
                    if item['tags']:
                        st.markdown(f"- **Tags:** {', '.join(item['tags'])} 🏷️")
                    st.markdown("---")
            else:
                st.info(f"😢 Puxa! Nenhum flashcard encontrado com o termo '{termo_busca}'. Que tal adicionar um novo? ➕")
        else:
            st.warning("Não se esqueça de digitar o que procurar no baú! 🧐")

# Opção: Adicionar Flashcard Manual
elif opcao_selecionada == "➕ Adicionar Flashcard Manual":
    st.header("Adicione um Novo Conhecimento! ✍️")
    st.info("Ajude a IA a ficar mais inteligente! Adicione seus próprios flashcards aqui. 🧠")
    
    with st.form("form_add_flashcard"):
        nova_pergunta = st.text_input("Qual a pergunta?", placeholder="Ex: Qual o teorema de Pitágoras?")
        nova_resposta = st.text_area("Qual a resposta mágica?", height=100, placeholder="Ex: Em um triângulo retângulo, o quadrado da hipotenusa é igual à soma dos quadrados dos catetos.")
        nova_imagem_url = st.text_input("URL da Imagem (opcional):", placeholder="Ex: https://upload.wikimedia.org/wikipedia/commons/4/4e/Pythagoras_cut.svg")
        novas_tags_str = st.text_input("Tags para organizar (separadas por vírgula, ex: matematica, algebra):", placeholder="Ex: história, geografia, biologia")
        
        submitted = st.form_submit_button("Adicionar este Saber! ✨")
        
        if submitted:
            if nova_pergunta and nova_resposta:
                novas_tags = [tag.strip() for tag in novas_tags_str.split(',') if tag.strip()]
                salvar_pergunta(nova_pergunta, nova_resposta, novas_tags, nova_imagem_url)
            else:
                st.error("Ops! 🛑 Por favor, preencha a pergunta E a resposta para adicionar o flashcard.")

# Modo Quiz (Múltipla Escolha por Tópico)
elif opcao_selecionada == "❓ Modo Quiz (Múltipla Escolha)":
    st.header("Modo Quiz: Múltipla Escolha por Tópico! 🧠💡")
    st.info("Escolha um tópico e teste seu conhecimento com questões de múltipla escolha! 🚀")

    banco_completo = carregar_perguntas()
    
    # Extrair todos os tópicos únicos
    todos_os_topicos = sorted(list(set(tag for item in banco_completo for tag in item.get('tags', []))))
    
    if not banco_completo:
        st.warning("Parece que seu banco de flashcards está vazio! 😔 Adicione alguns com tags para começar o quiz!")
    elif not todos_os_topicos:
        st.warning("Nenhum tópico encontrado nos seus flashcards! Por favor, adicione tags aos seus flashcards para usar o quiz por tópico.")
    else:
        # Seleção de tópico
        st.session_state.selected_topic = st.selectbox(
            "Selecione um Tópico para o Quiz:", 
            ["Todos os Tópicos"] + todos_os_topicos,
            key="topic_select"
        )
        
        # Filtra flashcards pelo tópico selecionado
        if st.session_state.selected_topic == "Todos os Tópicos":
            flashcards_filtrados = banco_completo
        else:
            flashcards_filtrados = [item for item in banco_completo if st.session_state.selected_topic in item.get('tags', [])]

        if not flashcards_filtrados or len(flashcards_filtrados) < NUM_ALTERNATIVAS: # Precisamos de pelo menos NUM_ALTERNATIVAS para gerar opções
            st.warning(f"Ops! Preciso de pelo menos {NUM_ALTERNATIVAS} flashcards no tópico '{st.session_state.selected_topic}' para criar um quiz de múltipla escolha. Adicione mais ou escolha 'Todos os Tópicos'!")
        else:
            # Inicialização do estado do quiz
            if 'quiz_data' not in st.session_state:
                st.session_state.quiz_data = {
                    'score': 0,
                    'question_index': 0,
                    'current_flashcard': None,
                    'options': [],
                    'correct_answer': ""
                }
            
            # --- Lógica para gerar a próxima questão ---
            def get_next_multiple_choice_question(flashcards_disponiveis):
                # Seleciona um flashcard aleatório para a pergunta
                current_flashcard = random.choice(flashcards_disponiveis)
                correct_answer = current_flashcard['resposta']
                
                # Coleta respostas de outros flashcards para usar como distratores
                # Tenta pegar do mesmo tópico primeiro para distratores mais plausíveis
                distractor_pool_same_topic = [
                    item['resposta'] for item in flashcards_disponiveis 
                    if item['resposta'] != correct_answer and item != current_flashcard
                ]
                
                # Se não houver suficientes no mesmo tópico, pega do banco completo
                # Filtra para que os distratores também sejam únicos e não a resposta correta
                distractor_pool_all = [
                    item['resposta'] for item in banco_completo 
                    if item['resposta'] != correct_answer and item['resposta'] not in distractor_pool_same_topic
                ]
                
                # Combine e remova duplicatas, priorizando do mesmo tópico
                distractor_pool = list(set(distractor_pool_same_topic + distractor_pool_all))

                # Seleciona distratores únicos (o número necessário)
                num_distractors_needed = NUM_ALTERNATIVAS - 1
                if len(distractor_pool) < num_distractors_needed:
                    # Não há distratores suficientes, avisa e tenta usar o que tem
                    st.warning(f"Não há distratores únicos suficientes no banco de dados para criar {NUM_ALTERNATIVAS} opções. Algumas opções podem se repetir ou não serem geradas.")
                    distractors = random.sample(distractor_pool, len(distractor_pool))
                else:
                    distractors = random.sample(distractor_pool, num_distractors_needed)
                
                # Adiciona a resposta correta às opções
                options = distractors + [correct_answer]
                random.shuffle(options) # Embaralha as opções
                
                st.session_state.quiz_data['current_flashcard'] = current_flashcard
                st.session_state.quiz_data['options'] = options
                st.session_state.quiz_data['correct_answer'] = correct_answer
                st.session_state.quiz_data['user_answer'] = None # Reseta a resposta do usuário
                st.session_state.quiz_data['answered_correctly'] = None # Reseta o status da resposta

            # --- Iniciar ou Continuar o Quiz ---
            if st.button("Iniciar Novo Quiz / Próxima Questão 👉"):
                get_next_multiple_choice_question(flashcards_filtrados)
                st.session_state.quiz_data['answered_correctly'] = None # Reseta ao pular ou iniciar
                st.rerun() # Força rerun para exibir a nova questão

            current_q = st.session_state.quiz_data['current_flashcard']

            if current_q:
                st.subheader(f"Questão {st.session_state.quiz_data['question_index'] + 1}:")
                st.markdown(f"**❓ Pergunta:** {current_q['pergunta']}")
                if current_q.get('imagem_url'):
                    st.image(current_q['imagem_url'], caption="Imagem da Questão", width=IMAGEM_LARGURA_PADRAO)

                # Opções de múltipla escolha
                # A seleção da opção só é processada se ainda não foi respondida
                if st.session_state.quiz_data['user_answer'] is None:
                    selected_option = st.radio(
                        "Escolha a resposta correta:",
                        st.session_state.quiz_data['options'],
                        index=None, # Inicia sem seleção
                        key=f"option_radio_{st.session_state.quiz_data['question_index']}" # Chave única para o radio button
                    )

                    if selected_option is not None: # Se o usuário selecionou uma opção
                        st.session_state.quiz_data['user_answer'] = selected_option
                        
                        if selected_option == st.session_state.quiz_data['correct_answer']:
                            st.session_state.quiz_data['score'] += 1
                            st.session_state.quiz_data['answered_correctly'] = True
                            st.success("🎉 Correto! Muito bem, Gênio! 🎉")
                        else:
                            st.session_state.quiz_data['answered_correctly'] = False
                            st.error(f"❌ Incorreto. A resposta certa era: **{st.session_state.quiz_data['correct_answer']}**")
                        st.rerun() # Força o rerun para mostrar o feedback

                # Exibe feedback após a resposta (se já foi respondida)
                if st.session_state.quiz_data['answered_correctly'] is not None:
                    if st.session_state.quiz_data['answered_correctly']:
                        st.success(f"Você acertou a questão {st.session_state.quiz_data['question_index'] + 1}!")
                    else:
                        st.error(f"Você errou a questão {st.session_state.quiz_data['question_index'] + 1}.")
                        st.info(f"A resposta correta é: **{st.session_state.quiz_data['correct_answer']}**")

                    if st.button("Próxima Questão 👉"):
                        st.session_state.quiz_data['question_index'] += 1
                        get_next_multiple_choice_question(flashcards_filtrados)
                        st.rerun()

                st.markdown(f"---")
                st.subheader(f"Pontuação Atual: {st.session_state.quiz_data['score']}")
                
            else:
                st.warning("Não foi possível carregar a questão. Verifique se o tópico selecionado tem flashcards suficientes.")


# Rodapé
st.sidebar.markdown("---")
st.sidebar.info("Feito com ❤️ por seu amigo(a) de estudos com IA! © 2025")
st.sidebar.markdown("Para uma experiência completa, garanta que todos os dados do NLTK estão baixados! 📚")