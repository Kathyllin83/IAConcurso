from nltk.chunk import ne_chunk

def gerar_flashcard_avancado(texto):
    sentencas = sent_tokenize(texto)
    if not sentencas:
        return None, None

    primeira_sentenca = sentencas[0]
    palavras = word_tokenize(primeira_sentenca)
    tags = pos_tag(palavras)

    # Tentar reconhecimento de entidades nomeadas
    named_entities = []
    tree = ne_chunk(tags)
    for subtree in tree.subtrees():
        if subtree.label() in ['PERSON', 'ORGANIZATION', 'LOCATION', 'GPE', 'DATE', 'TIME', 'MONEY', 'PERCENT']:
            named_entities.append(" ".join([leaf[0] for leaf in subtree.leaves()]))

    # Lógica de geração de pergunta melhorada
    pergunta = ""
    resposta = primeira_sentenca

    if named_entities:
        # Prioriza entidades nomeadas para a pergunta
        principal_entidade = named_entities[0]
        
        # Regras baseadas na entidade
        if principal_entidade.lower() in [e.lower() for e, t in tags if t.startswith('NNP')]: # Se for um nome próprio
             if 'é' in palavras or 'são' in palavras:
                 pergunta = f"Quem é {principal_entidade}?" if 'PERSON' in [subtree.label() for subtree in tree.subtrees() if principal_entidade in " ".join([leaf[0] for leaf in subtree.leaves()])] else f"O que é {principal_entidade}?"
             elif 'onde' in primeira_sentenca.lower() or 'localizado' in primeira_sentenca.lower() and 'LOCATION' in [subtree.label() for subtree in tree.subtrees() if principal_entidade in " ".join([leaf[0] for leaf in subtree.leaves()])]:
                 pergunta = f"Onde está {principal_entidade}?"
             elif 'quando' in primeira_sentenca.lower() or 'data' in primeira_sentenca.lower() and 'DATE' in [subtree.label() for subtree in tree.subtrees() if principal_entidade in " ".join([leaf[0] for leaf in subtree.leaves()])]:
                 pergunta = f"Quando {principal_entidade}?"
             else:
                 pergunta = f"Fale sobre {principal_entidade}."

        elif 'é' in palavras or 'são' in palavras:
            substantivos = [word for word, tag in tags if tag.startswith('NN')]
            if substantivos:
                pergunta = f"O que é {substantivos[0]}?"
            else:
                pergunta = f"Qual a ideia principal sobre {principal_entidade}?"
        else:
            pergunta = f"Qual a informação principal sobre: '{principal_entenca}'?" # Fallback
            
    else: # Fallback para o caso de não encontrar entidades nomeadas
        substantivos = [word for word, tag in tags if tag.startswith('NN')]
        if substantivos:
            pergunta = f"O que é {substantivos[0]}?"
        elif 'é' in palavras or 'são' in palavras:
            pergunta = f"O que está sendo definido em: '{primeira_sentenca}'?"
        else:
            pergunta = f"Qual a ideia principal de: '{primeira_sentenca}'?"

    return pergunta, resposta