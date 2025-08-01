from flask import Flask, render_template, request, redirect
from database import get_db_connection  # Importa a função de conexão com o banco de dados

# Cria uma instância da aplicação Flask
app = Flask(__name__)


# Rota principal que exibe o menu
@app.route('/')
def index():
    return render_template('menu.html')  # Renderiza o template do menu principal


# Rota para cadastro de pacientes
@app.route('/cadpaciente', methods=['GET', 'POST'])
def cadastro():
    # Conecta ao banco de dados e obtém a lista de planos de saúde
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT codPlano, nome FROM planosaude")
    planos = cursor.fetchall()

    if request.method == 'POST':
        # Obtém e trata os dados do formulário
        nome = request.form['nome'].strip() #.strip() retira espaços desnecessários antes e depois da cadeia da caracteres
        cpf = request.form['cpf'].strip()
        dataNasc = request.form['dataNasc']
        codPlano = request.form['codPlano']

        try: # Tenta rodar oque está dentro
            # Tenta inserir o novo paciente no banco de dados
            cursor.execute(
                "INSERT INTO paciente (nome, cpf, dataNasc, codPlano) VALUES (%s, %s, %s, %s)",
                (nome, cpf, dataNasc, codPlano)
            )
            conn.commit()
            mensagem = "Paciente cadastrado com sucesso!" # Se conseguiu inserir o novo registro mostra a mensagem de sucesso

        except Exception as e: # Exceto se houver um erro
            # Em caso de erro, faz rollback e mostra mensagem de erro
            conn.rollback() # rollback desfaz todas as alterações não confirmadas
            mensagem = f"Erro ao cadastrar: {str(e)}"

        finally:
            # Retorna o template com a mensagem apropriada
            return render_template('cadpaciente.html',
                               mensagem_sucesso=mensagem,
                               planos_saude=planos)

    # Fecha a conexão e retorna o template para requisições GET
    conn.close()
    return render_template('cadpaciente.html', planos_saude=planos)


# Rota para cadastro de planos de saúde
@app.route('/cadplano', methods=['GET', 'POST'])
def cadplano():
    # Verifica se a requisição é do tipo POST (envio de formulário)
    if request.method == 'POST':
        # Obtém e valida o nome do plano
        nome = request.form.get('nome', '').strip()
        # Verifica se o nome do plano foi fornecido
        if not nome:
            # Retorna o template com mensagem de erro se o nome estiver vazio
            return render_template('cadplano.html',
                                   mensagem_erro="Nome do plano é obrigatório")

        # Conecta ao banco de dados
        conn = get_db_connection()
        # Cria um cursor para executar comandos SQL
        cursor = conn.cursor()

        try:
            # Tenta inserir o novo plano no banco de dados
            # Executa a query SQL com parâmetro seguro (%s)
            cursor.execute(
                "INSERT INTO planosaude (nome) VALUES (%s)",
                (nome,)  # Tupla com o valor do parâmetro
            )
            # Confirma a transação no banco de dados
            conn.commit()
            # Define mensagem de sucesso
            mensagem = f"Plano cadastrado com sucesso!"

        except Exception as e:
            # Em caso de erro, faz rollback (Desfaz qualquer alteração não confirmada no banco)
            conn.rollback()
            # Cria mensagem de erro com a descrição da exceção
            mensagem = f"Erro ao cadastrar: {str(e)}"

        finally:
            # Fecha a conexão com o banco de dados
            conn.close()
            # Retorna o template com a mensagem apropriada
            return render_template('cadplano.html',
                                   mensagem_sucesso=mensagem,
                                   nome_plano=nome)

    # Retorna o template para requisições GET (acesso normal à página)
    return render_template('cadplano.html')


# Rota para cadastro de especialidades médicas
@app.route('/cadespec', methods=['GET', 'POST'])
def cadespec():
    if request.method == 'POST':
        # Obtém e valida o nome da especialidade
        nome = request.form.get('nome', '').strip()
        if not nome:
            return render_template('cadespec.html',
                                   mensagem_erro="Nome da especialidade é obrigatório")

        # Conecta ao banco de dados
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # Tenta inserir a nova especialidade no banco
            cursor.execute(
                "INSERT INTO especialidade (nome) VALUES (%s)",
                (nome,)
            )
            conn.commit()
            mensagem = f"Especialidade cadastrada com sucesso!"

        except Exception as e:
            # Em caso de erro, faz rollback
            conn.rollback()
            mensagem = f"Erro ao cadastrar: {str(e)}"

        finally:
            # Fecha a conexão e retorna o template
            conn.close()
            return render_template('cadespec.html',
                                   mensagem_sucesso=mensagem,
                                   nome_plano=nome)

    # Retorna o template para requisições GET
    return render_template('cadespec.html')


# Rota para cadastro de médicos
@app.route('/cadmedico', methods=['GET', 'POST'])
def cadmedico():
    # Conecta ao banco e obtém a lista de especialidades
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT codespec, nome FROM especialidade")
    especialidades = cursor.fetchall()

    if request.method == 'POST':
        # Obtém os dados do formulário
        nome = request.form['nome'].strip()
        CRM = request.form['CRM'].strip()
        codespec = request.form['codespec']

        try:
            # Tenta inserir o novo médico no banco
            cursor.execute(
                "INSERT INTO medico (nome, CRM, codespec) VALUES (%s, %s, %s)",
                (nome, CRM, codespec)
            )
            conn.commit()
            mensagem = "Médico cadastrado com sucesso!"

        except Exception as e:
            # Em caso de erro, faz rollback
            conn.rollback()
            mensagem = f"Erro ao cadastrar: {str(e)}"

        finally:
            # Fecha a conexão e retorna o template
            conn.close()
            return render_template('cadmedico.html',
                                   espec=especialidades,
                                   mensagem_sucesso=mensagem)

    # Fecha a conexão e retorna o template para requisições GET
    conn.close()
    return render_template('cadmedico.html', espec=especialidades)


# Rota para consulta de pacientes cadastrados
@app.route('/consulta')
def consulta():
    try:
        # Conecta ao banco
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query que seleciona os dados da tabela paciente e o nome do plano de saúde da tabela planosaude
        cursor.execute("""
            SELECT p.nome, p.cpf, p.dataNasc, ps.nome as nome_plano
            FROM Paciente p LEFT JOIN planosaude ps ON p.codPlano = ps.codPlano
        """)
        # Busca todos os pacientes
        pacientes = cursor.fetchall()

        # Fecha a conexão e retorna o template com os pacientes
        conn.close()
        return render_template('consulta.html', pacientes=pacientes)

    except Exception as e:
        # Em caso de erro, retorna mensagem de erro
        return render_template('consulta.html',
                               mensagem_erro=f"Erro ao carregar pacientes: {str(e)}")


# Ponto de entrada da aplicação
if __name__ == '__main__':
    # Inicia o servidor Flask em modo de desenvolvimento
    app.run(debug=True)
