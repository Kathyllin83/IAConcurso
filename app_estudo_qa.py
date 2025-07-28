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
import matplotlib.pyplot as plt 


# --- Configurações Globais ---
DATA_FILE = 'banco_de_perguntas.json'
QUIZ_HISTORY_FILE = 'quiz_history.json' 
LIMIAR_SIMILARIDADE = 0.25 
IMAGEM_LARGURA_PADRAO = 200 
NUM_ALTERNATIVAS = 4 

# --- Caminho Local do Logo ---
LOCAL_LOGO_PATH = "logo.png" 

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
        "imagem_url": imagem_url if imagem_url else "" 
    }
    banco_dados.append(novo_item)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(banco_dados, f, indent=4, ensure_ascii=False)
    st.balloons() 
    st.success("🎉 Flashcard salvo com sucesso no seu cérebro digital! 🎉")


def buscar_pergunta_existente(termo_busca):
    """
    Busca no banco de dados por perguntas ou respostas que contenham o termo.
    Retorna uma lista de itens que correspondem.
    """
    termo_busca_lower = termo_busca.lower()
    resultados = []
    banco_dados = carregar_perguntas() 
    for item in banco_dados:
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
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        st.info("Parece que as 'stopwords' do NLTK (para português) ainda não estão por aqui... Baixando para você! 🚀")
        nltk.download('stopwords')

    textos_para_vetorizar = [item['pergunta'] for item in banco_dados]

    if not textos_para_vetorizar:
        return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('portuguese')), np.array([]), []

    vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('portuguese'), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(textos_para_vetorizar)
    
    return vectorizer, tfidf_matrix, banco_dados

vectorizer, tfidf_matrix, banco_dados_qa = preencher_e_vetorizar_banco()


def responder_pergunta_qa(pergunta_usuario):
    """
    Busca a resposta mais relevante no banco de dados usando similaridade TF-IDF e Cosseno.
    """
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

# --- Funções Auxiliares para o Quiz (Gráfico) ---
def plot_pie_chart(correct_percentage, error_percentage, title='Desempenho'):
    """Gera e exibe um gráfico de pizza com as porcentagens de acertos e erros."""
    labels = ['Acertos', 'Erros']
    sizes = [correct_percentage, error_percentage]
    colors = ['#4CAF50', '#F44336'] # Verde para acertos, Vermelho para erros
    explode = (0.05, 0) # Destaca a fatia de 'Acertos'

    fig1, ax1 = plt.subplots(figsize=(1.5, 1.5)) # Tamanho bastante compacto
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'color': 'white', 'fontsize': 6}) # Fonte menor para caber
    ax1.axis('equal')  
    ax1.set_title(title, color='white', fontsize=7) # Título menor

    fig1.patch.set_facecolor('None') 
    ax1.set_facecolor('None') 
    
    col_grafico_center = st.columns([1, 0.5, 1])[1] # Coluna do meio para o gráfico, com 0.5 de proporção
    with col_grafico_center:
        st.pyplot(fig1) 
    plt.close(fig1) 

# --- Funções de Persistência do Histórico do Quiz ---
def salvar_historico_quiz():
    """Salva o histórico de quizzes e os dados do quiz atual em um arquivo JSON."""
    data_to_save = {
        'quiz_history': st.session_state.quiz_history,
        'current_quiz_data': st.session_state.current_quiz_data
    }
    with open(QUIZ_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

def carregar_historico_quiz():
    """
    Carrega o histórico de quizzes e os dados do quiz atual de um arquivo JSON.
    Esta função deve ser chamada APENAS UMA VEZ na primeira execução do script.
    """
    # NOVO: Verifica se a função já foi executada para evitar re-carregamento desnecessário
    if 'history_loaded' not in st.session_state:
        st.session_state.history_loaded = True # Marca que o carregamento inicial ocorreu

        # Inicializa com valores padrão. Estes serão sobrescritos se o arquivo existir.
        st.session_state.quiz_history = []
        st.session_state.current_quiz_data = {
            'quiz_name': 'Quiz Atual',
            'score': 0,
            'question_index': 0,
            'current_flashcard': None,
            'options': [],
            'correct_answer': "",
            'total_answered': 0,
            'correct_count': 0,
            'quiz_started': False,
            'details': [],
            'selected_option': None, 
            'quiz_submitted': False 
        }

        if os.path.exists(QUIZ_HISTORY_FILE):
            try:
                with open(QUIZ_HISTORY_FILE, 'r', encoding='utf-8') as f:
                    data_loaded = json.load(f)
                    
                    if 'quiz_history' in data_loaded:
                        st.session_state.quiz_history = data_loaded['quiz_history']
                    if 'current_quiz_data' in data_loaded:
                        loaded_data = data_loaded['current_quiz_data']
                        # Garante que todas as chaves esperadas existam, mesmo se não estiverem no arquivo antigo
                        loaded_data.setdefault('selected_option', None)
                        loaded_data.setdefault('quiz_submitted', False)
                        st.session_state.current_quiz_data = loaded_data
            except json.JSONDecodeError:
                st.warning("Arquivo de histórico do quiz corrompido. Iniciando um novo histórico.")
            except Exception as e:
                st.error(f"Erro ao carregar histórico do quiz: {e}. Iniciando um novo histórico.")
            

# --- NOVO: Função Global para Obter a Próxima Questão do Quiz ---
# Esta função foi movida para o escopo global para evitar NameError
def get_next_multiple_choice_question(flashcards_disponiveis, banco_completo):
    """
    Seleciona e prepara a próxima questão para o quiz.
    Atualiza st.session_state.current_quiz_data com os dados da nova questão.
    """
    # Só busca uma nova questão se a resposta da anterior já foi submetida ou se é a primeira questão
    if st.session_state.current_quiz_data['quiz_submitted'] or \
       st.session_state.current_quiz_data['current_flashcard'] is None: 
        
        # Certifica-se de que há flashcards disponíveis antes de tentar escolher
        if not flashcards_disponiveis:
            # Não use 'return' aqui se estiver fora da função ou se o Streamlit espera uma renderização completa.
            # Em vez de 'return', ajustamos o estado para indicar que não há questões.
            st.warning("Não há flashcards disponíveis para iniciar o quiz com este tópico/filtros. Adicione mais flashcards com tags ou escolha 'Todos os Tópicos'.")
            st.session_state.current_quiz_data['quiz_started'] = False # Marca o quiz como não iniciado
            st.session_state.current_quiz_data['current_flashcard'] = None # Limpa a questão atual
            st.session_state.current_quiz_data['quiz_submitted'] = False # Assegura que o quiz esteja limpo
            return # AQUI: O return está dentro da função 'get_next_multiple_choice_question', está correto.
            
        current_flashcard = random.choice(flashcards_disponiveis)
        correct_answer = current_flashcard['resposta']
        
        distractor_pool_same_topic = [
            item['resposta'] for item in flashcards_disponiveis 
            if item['resposta'] != correct_answer and item != current_flashcard
        ]
        
        distractor_pool_all = [
            item['resposta'] for item in banco_completo 
            if item['resposta'] != correct_answer and item['resposta'] not in distractor_pool_same_topic
        ]
        
        distractor_pool = list(set(distractor_pool_same_topic + distractor_pool_all))

        num_distractors_needed = NUM_ALTERNATIVAS - 1
        # Pega o mínimo entre o que precisa e o que tem para distratores
        num_to_sample = min(num_distractors_needed, len(distractor_pool))
        distractors = random.sample(distractor_pool, num_to_sample)
        
        options = distractors + [correct_answer]
        random.shuffle(options) 
        
        st.session_state.current_quiz_data['current_flashcard'] = current_flashcard
        st.session_state.current_quiz_data['options'] = options
        st.session_state.current_quiz_data['correct_answer'] = correct_answer
        st.session_state.current_quiz_data['user_answer'] = None 
        st.session_state.current_quiz_data['answered_correctly'] = None 
        st.session_state.current_quiz_data['quiz_submitted'] = False # Reseta a flag de submissão para a nova questão
        
        # Incrementa o question_index APENAS quando uma NOVA questão é carregada
        # (e se o current_flashcard foi de fato setado, ou seja, se a lista não estava vazia)
        if st.session_state.current_quiz_data['current_flashcard'] is not None:
            st.session_state.current_quiz_data['question_index'] += 1


# --- Interface do Streamlit ---

st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Cérebro de Pão", page_icon="🍞")

st.title("🧠 Cérebro de Pão! 🍞")
st.markdown("Bem-vindo(a) à sua ferramenta de estudo superpotente! 🚀")

if os.path.exists(LOCAL_LOGO_PATH):
    st.sidebar.image(LOCAL_LOGO_PATH, width=150)
else:
    st.sidebar.warning("Logo não encontrado! Verifique o caminho: logo.png")

st.sidebar.markdown("---") 

st.sidebar.header("Escolha sua Aventura! 🗺️")

opcao_selecionada = st.sidebar.radio(
    "O que vamos aprender hoje?",
    ("🗣️ Fazer uma Pergunta à IA", "📝 Gerar Flashcard (IA Básica)", "📚 Consultar Flashcards", 
     "➕ Adicionar Flashcard Manual", "⭐ Iniciar Novo Quiz", "❓ Modo Quiz (Múltipla Escolha)", "📈 Desempenho do Quiz") 
)

# --- CHAME A FUNÇÃO DE CARREGAMENTO AQUI, NO INÍCIO DO SCRIPT ---
# A função agora tem uma flag interna para garantir que só carrega na primeira execução.
# Mover esta chamada para dentro de um "if 'history_loaded' not in st.session_state:"
# que está dentro da função 'carregar_historico_quiz()' já é o controle que precisamos.
carregar_historico_quiz() 

# --- SEÇÕES DO APP ---

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


# --- INÍCIO DA SEÇÃO: INICIAR NOVO QUIZ (para definir o nome do quiz) ---
elif opcao_selecionada == "⭐ Iniciar Novo Quiz":
    st.header("Hora de Criar um Novo Quiz! ✨")
    st.info("Dê um nome ao seu desafio de hoje e vamos começar a aprender! 📝")

    default_quiz_name = f"Meu Quiz {len(st.session_state.quiz_history) + 1}"
    if st.session_state.current_quiz_data['quiz_started'] and st.session_state.current_quiz_data['quiz_name'] != 'Quiz Atual':
        default_quiz_name = st.session_state.current_quiz_data['quiz_name']


    quiz_name_input = st.text_input(
        "Nome do seu Quiz:", 
        value=default_quiz_name, 
        key="quiz_name_setter"
    )

    if st.button("Começar Este Quiz! 🚀"):
        if quiz_name_input.strip():
            st.session_state.current_quiz_data = {
                'quiz_name': quiz_name_input.strip(),
                'score': 0,
                'question_index': 0, 
                'current_flashcard': None,
                'options': [],
                'correct_answer': "",
                'total_answered': 0,
                'correct_count': 0,
                'quiz_started': True, 
                'details': [],
                'selected_option': None, 
                'quiz_submitted': False 
            }
            salvar_historico_quiz() 
            st.success(f"Quiz '{quiz_name_input}' pronto para começar! Agora, vá para a aba '❓ Modo Quiz (Múltipla Escolha)' para jogar!")
            st.rerun() 
        else:
            st.warning("Por favor, digite um nome para o seu quiz antes de começar! 😅")

# --- FIM DA SEÇÃO: INICIAR NOVO QUIZ ---


# --- INÍCIO DA SEÇÃO "MODO QUIZ" (A LÓGICA DO QUIZ AGORA COM ST.FORM) ---
elif opcao_selecionada == "❓ Modo Quiz (Múltipla Escolha)":
    st.header(f"Modo Quiz: {st.session_state.current_quiz_data['quiz_name']}! 🧠💡")
    
    if not st.session_state.current_quiz_data['quiz_started']:
        st.info("Você precisa iniciar um novo quiz na aba '⭐ Iniciar Novo Quiz' antes de começar a jogar. 🚀")
    else:
        st.info("Hora de testar seus conhecimentos com questões de múltipla escolha! 🤓")

        banco_completo = carregar_perguntas()
        
        todos_os_topicos = sorted(list(set(tag for item in banco_completo for tag in item.get('tags', []))))
        
        # Seleção de tópico
        st.session_state.selected_topic = st.selectbox(
            "Selecione um Tópico para o Quiz:", 
            ["Todos os Tópicos"] + todos_os_topicos,
            key="topic_select_quiz_mode"
        )
        
        if st.session_state.selected_topic == "Todos os Tópicos":
            flashcards_filtrados = banco_completo
        else:
            flashcards_filtrados = [item for item in banco_completo if st.session_state.selected_topic in item.get('tags', [])]

        # Verifica se há flashcards disponíveis antes de prosseguir
        if not flashcards_filtrados or len(flashcards_filtrados) < NUM_ALTERNATIVAS:
            st.warning(f"Ops! Preciso de pelo menos {NUM_ALTERNATIVAS} flashcards no tópico '{st.session_state.selected_topic}' para criar um quiz de múltipla escolha. Adicione mais ou escolha 'Todos os Tópicos'!")
            # Interrompe o quiz se não há questões válidas
            st.session_state.current_quiz_data['quiz_started'] = False
            st.session_state.current_quiz_data['current_flashcard'] = None
            st.session_state.current_quiz_data['quiz_submitted'] = False # Assegura que o quiz esteja limpo
            # NÃO USAR RETURN AQUI. O resto do bloco 'else' será pulado se quiz_started for False.
        else: # Somente entra neste 'else' se há flashcards suficientes e o quiz está 'started'
            # NOVO: Chamada da função get_next_multiple_choice_question aqui, no início do fluxo do quiz
            # para garantir que a questão esteja sempre carregada corretamente.
            get_next_multiple_choice_question(flashcards_filtrados, banco_completo) 
            
            current_q = st.session_state.current_quiz_data['current_flashcard']

            if current_q: # Continua apenas se uma questão foi carregada com sucesso
                st.subheader(f"Questão {st.session_state.current_quiz_data['question_index']}:") 
                st.markdown(f"**❓ Pergunta:** {current_q['pergunta']}")
                if current_q.get('imagem_url'):
                    st.image(current_q['imagem_url'], caption="Imagem da Questão", width=IMAGEM_LARGURA_PADRAO)

                # --- ENVOLVENDO A LÓGICA DE RESPOSTA EM UM ST.FORM ---
                with st.form(key=f"quiz_form_{st.session_state.current_quiz_data['question_index']}"):
                    selected_option = st.radio(
                        "Escolha a resposta correta:",
                        st.session_state.current_quiz_data['options'],
                        index=st.session_state.current_quiz_data['options'].index(st.session_state.current_quiz_data['user_answer']) if st.session_state.current_quiz_data['user_answer'] else None,
                        disabled=st.session_state.current_quiz_data['quiz_submitted'], # Desabilita após submissão
                        key=f"radio_{st.session_state.current_quiz_data['question_index']}" 
                    )
                    
                    # O botão de submissão do formulário
                    submitted = st.form_submit_button("Verificar Resposta ✅", disabled=st.session_state.current_quiz_data['quiz_submitted'])
                    
                    if submitted:
                        if selected_option is not None:
                            st.session_state.current_quiz_data['user_answer'] = selected_option
                            st.session_state.current_quiz_data['total_answered'] += 1 
                            
                            is_correct = (selected_option == st.session_state.current_quiz_data['correct_answer'])

                            st.session_state.current_quiz_data['details'].append({
                                'pergunta': current_q['pergunta'],
                                'resposta_correta': st.session_state.current_quiz_data['correct_answer'],
                                'resposta_usuario': selected_option, 
                                'status': 'Certo' if is_correct else 'Errado'
                            })
                            
                            if is_correct:
                                st.session_state.current_quiz_data['score'] += 1
                                st.session_state.current_quiz_data['answered_correctly'] = True
                                st.success("🎉 Correto! Muito bem, Gênio! 🎉")
                            else:
                                st.session_state.current_quiz_data['answered_correctly'] = False
                                st.error(f"❌ Incorreto. A resposta certa era: **{st.session_state.current_quiz_data['correct_answer']}**")
                                st.info(f"A resposta correta era: **{st.session_state.current_quiz_data['correct_answer']}**") 
                            
                            st.session_state.current_quiz_data['correct_count'] = sum(1 for d in st.session_state.current_quiz_data['details'] if d['status'] == 'Certo')
                            st.session_state.current_quiz_data['quiz_submitted'] = True # MARCA QUE A RESPOSTA FOI SUBMETIDA
                            salvar_historico_quiz() 
                            st.rerun() 
                        else:
                            st.warning("Por favor, selecione uma resposta antes de verificar! 😉")

            # Exibir feedback e botão "Próxima Questão" APÓS A RESPOSTA SER SUBMETIDA
            if st.session_state.current_quiz_data['quiz_submitted']:
                # Re-exibe o feedback para garantir que seja visto
                if st.session_state.current_quiz_data['answered_correctly']:
                    st.success("🎉 Correto! Muito bem, Gênio! 🎉")
                else:
                    st.error(f"❌ Incorreto. A resposta certa era: **{st.session_state.current_quiz_data['correct_answer']}**")
                    st.info(f"A resposta correta era: **{st.session_state.current_quiz_data['correct_answer']}**")
                
                if st.button("Próxima Questão 👉", key="next_question_btn"):
                    st.session_state.current_quiz_data['quiz_submitted'] = False # Reseta a flag para a próxima questão
                    # A função get_next_multiple_choice_question será chamada novamente no topo do loop
                    # para pegar a nova questão.
                    salvar_historico_quiz() 
                    st.rerun()

            st.markdown(f"---")
            st.subheader(f"Pontuação Atual: {st.session_state.current_quiz_data['score']}")
            
            if st.session_state.current_quiz_data['total_answered'] > 0:
                correct_percentage = (st.session_state.current_quiz_data['correct_count'] / st.session_state.current_quiz_data['total_answered']) * 100
                error_percentage = 100 - correct_percentage
                
                st.markdown(f"**Total de Perguntas Respondidas:** {st.session_state.current_quiz_data['total_answered']}")
                st.markdown(f"**Acertos:** {st.session_state.current_quiz_data['correct_count']}")
                st.markdown(f"**% de Acertos:** :green[{correct_percentage:.2f}%]")
                st.markdown(f"**% de Erros:** :red[{error_percentage:.2f}%]")
            else:
                st.info("Responda à primeira pergunta para ver seu desempenho!")
            
            st.markdown("---")
            # Botão para finalizar o quiz atual e salvar no histórico
            if st.button("Finalizar este Quiz e Salvar Resultados", key="finish_quiz_btn_bottom"): 
                if st.session_state.current_quiz_data['total_answered'] > 0:
                    st.session_state.quiz_history.append(st.session_state.current_quiz_data.copy())
                    st.success(f"Quiz '{st.session_state.current_quiz_data['quiz_name']}' finalizado e salvo no histórico! Vá para a aba '📈 Desempenho do Quiz' para ver.")
                else:
                    st.warning("Quiz vazio, não há resultados para salvar.")
                
                st.session_state.current_quiz_data = {
                    'quiz_name': 'Quiz Atual',
                    'score': 0,
                    'question_index': 0,
                    'current_flashcard': None,
                    'options': [],
                    'correct_answer': "",
                    'total_answered': 0,
                    'correct_count': 0,
                    'quiz_started': False,
                    'details': [],
                    'selected_option': None,
                    'quiz_submitted': False 
                }
                salvar_historico_quiz() 
                st.rerun() 

            else: # Este else corresponde ao 'if current_q:'
                st.warning("Não foi possível carregar a questão. Verifique se o tópico selecionado tem flashcards suficientes.")
# --- FIM DA SEÇÃO "MODO QUIZ" ---


# --- INÍCIO DA SEÇÃO "DESEMPENHO DO QUIZ" ---
elif opcao_selecionada == "📈 Desempenho do Quiz":
    st.header("Seu Histórico de Desempenho no Quiz! 🚀")
    st.info("Confira como você está se saindo nos seus estudos em cada quiz que fez! 👇")

    # Verifica se há quizzes no histórico ou se o quiz atual está em andamento com dados
    # Esta condição foi reavaliada para ser mais simples e clara.
    if not st.session_state.quiz_history and \
       (not st.session_state.current_quiz_data['quiz_started'] or st.session_state.current_quiz_data['total_answered'] == 0):
        st.warning("Nenhum quiz concluído ou em andamento para exibir no histórico.")
    else:
        # Exibe o desempenho do quiz atual (se houver um em andamento e com perguntas respondidas)
        if st.session_state.current_quiz_data['quiz_started'] and st.session_state.current_quiz_data['total_answered'] > 0:
            st.subheader(f"Desempenho do Quiz Atual: '{st.session_state.current_quiz_data['quiz_name']}'")
            total_answered = st.session_state.current_quiz_data['total_answered']
            correct_count = st.session_state.current_quiz_data['correct_count']
            score = st.session_state.current_quiz_data['score']

            correct_percentage = (correct_count / total_answered) * 100
            error_percentage = 100 - correct_percentage

            st.markdown(f"- **Pontuação Total:** {score} pontos")
            st.markdown(f"- **Total de Perguntas Respondidas:** {total_answered}")
            st.markdown(f"- **Total de Acertos:** {correct_count}")
            st.markdown(f"- **Total de Erros:** {total_answered - correct_count}")
            st.markdown(f"- **% de Acertos:** :green[{correct_percentage:.2f}%]")
            st.markdown(f"**% de Erros:** :red[{error_percentage:.2f}%]")
            plot_pie_chart(correct_percentage, error_percentage, title='Quiz Atual') 
            
            if st.session_state.current_quiz_data['details']:
                with st.expander("Revisar Respostas do Quiz Atual"):
                    for idx, detail in enumerate(st.session_state.current_quiz_data['details']):
                        st.markdown(f"**Questão {idx+1}:**")
                        st.markdown(f"   **P:** {detail['pergunta']}")
                        st.markdown(f"   **Sua Resposta:** :blue[{detail['resposta_usuario']}]")
                        st.markdown(f"   **Correta:** :green[{detail['resposta_correta']}]")
                        if detail['status'] == 'Certo':
                            st.markdown(f"   **Status:** :green[Certa!] ✅")
                        else:
                            st.markdown(f"   **Status:** :red[Errada!] ❌")
                        st.markdown("---")

            st.markdown("---")


        if st.session_state.quiz_history:
            st.subheader("Quizzes Anteriores:")
            for i, quiz_result in enumerate(reversed(st.session_state.quiz_history)):
                idx_original = len(st.session_state.quiz_history) - 1 - i
                expander_title = f"Histórico #{idx_original + 1}: '{quiz_result['quiz_name']}' - Pontuação: {quiz_result['score']}"
                
                with st.expander(expander_title):
                    total_answered = quiz_result['total_answered']
                    correct_count = quiz_result['correct_count']
                    score = quiz_result['score']

                    if total_answered > 0:
                        correct_percentage = (correct_count / total_answered) * 100
                        error_percentage = 100 - correct_percentage
                        
                        st.markdown(f"- **Total de Perguntas Respondidas:** {total_answered}")
                        st.markdown(f"- **Total de Acertos:** {correct_count}")
                        st.markdown(f"- **Total de Erros:** {total_answered - correct_count}")
                        st.markdown(f"- **% de Acertos:** :green[{correct_percentage:.2f}%]")
                        st.markdown(f"**% de Erros:** :red[{error_percentage:.2f}%]")
                        plot_pie_chart(correct_percentage, error_percentage, title=f"Desempenho de '{quiz_result['quiz_name']}'")
                        
                        if quiz_result['details']:
                            st.markdown("---")
                            st.markdown("**Revisão de Respostas:**")
                            for idx, detail in enumerate(quiz_result['details']):
                                st.markdown(f"**Questão {idx+1}:**")
                                st.markdown(f"   **P:** {detail['pergunta']}")
                                st.markdown(f"   **Sua Resposta:** :blue[{detail['resposta_usuario']}]")
                                st.markdown(f"   **Correta:** :green[{detail['resposta_correta']}]")
                                if detail['status'] == 'Certo':
                                    st.markdown(f"   **Status:** :green[Certa!] ✅")
                                else:
                                    st.markdown(f"   **Status:** :red[Errada!] ❌")
                                st.markdown("---")

                    else:
                        st.info("Este quiz foi salvo, mas nenhuma pergunta foi respondida.")
                st.markdown("---") 
        
        # Botão para limpar todo o histórico (agora dentro do 'else' para ser exibido só se há histórico)
        if st.button("🗑️ Limpar TODO o Histórico de Quizzes"):
            st.session_state.quiz_history = []
            st.session_state.current_quiz_data = {
                'quiz_name': 'Quiz Atual',
                'score': 0,
                'question_index': 0,
                'current_flashcard': None,
                'options': [],
                'correct_answer': "",
                'total_answered': 0,
                'correct_count': 0,
                'quiz_started': False,
                'details': [],
                'selected_option': None,
                'quiz_submitted': False 
            }
            salvar_historico_quiz() 
            st.success("Histórico de quizzes limpo com sucesso!")
            st.rerun()

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info("Feito com ❤️ por seu amigo(a) de estudos com IA! © 2025")
st.sidebar.markdown("Para uma experiência completa, garanta que todos os dados do NLTK estão baixados! 📚")