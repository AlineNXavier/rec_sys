import streamlit as st
import pickle
from model_bert import BertModel
import pandas as pd
import numpy as np
from gensim.models import Word2Vec


class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def cal_percent(query_sentence, idx, df):
    query_sentence = query_sentence.split()
    count = 0
    palavras = []
    for p in query_sentence:
        if p in df.loc[idx, "Resumo"].lower():
            count = count + 1
            palavras.append(p)
            
    return count/(len(query_sentence)), palavras

def is_word_in_model(word, model):
    """
    Check on individual words ``word`` that it exists in ``model``.
    """
    assert type(model).__name__ == 'KeyedVectors'
    is_in_vocab = word in model.key_to_index.keys()
    return is_in_vocab


def main():
    
    with open(r"bert_model.pkl", "rb") as input_file:
        bert_model = pickle.load(input_file)
    
    st.title("Recomendação de Trabalhos Cientificos")

    df = pd.read_csv('dados_df2.csv',index_col='Unnamed: 0')


    # Adicione componentes ao seu aplicativo
    valor_usuario = st.text_input("Entre com uma frase:")

    if st.button("Pesquisar Recomendações", key='bt_rec'):
        # Ação a ser executada quando o botão "Pesquisar Recomendações" for clicado
        
        st.write("Resultado para o modelo BERT")

        indices, scr = bert_model.predict(valor_usuario)
        novos_resultados = df[['Resumo', 'Autor', 'Titulo', 'Tipo']].iloc[indices]

        novos_resultados.reset_index(inplace=True)
        novos_resultados['Palavras'] = novos_resultados.apply(lambda x: cal_percent(valor_usuario, x['ID'], df)[1], axis=1)
        novos_resultados['Score'] = scr

        st.write(novos_resultados[['ID', 'Autor', 'Titulo', 'Tipo', 'Palavras', 'Score','Resumo']])


    #if st.checkbox('Ver resumos'):
    #    id_resumo = st.text_input("ID do resumo:")

    #    if st.button("Pesquisar Resumo", key='bt_resumo'):
            # Ação a ser executada quando o botão "Pesquisar Resumo" for clica
         #  st.write(list(df[df.index == int(id_resumo)]['resumo']))

# Run the app
if __name__ == '__main__':
    main()

            
            
