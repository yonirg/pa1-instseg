# AI_LOG.md — como usamos IA neste assignment

**Resumo honesto:** o esqueleto deste repositório (estrutura de pacotes, primeiras versões
dos módulos, scripts de cada parte, README) foi gerado com o Claude (Anthropic) a partir
do PDF do PA, e depois rodado e depurado em conversa. Os experimentos reportados em
`RESULTS.md` foram executados nesse mesmo ambiente (1 núcleo de CPU) — por isso os
orçamentos de tempo curtos. Tudo o que está aqui nós lemos, entendemos e conseguimos
explicar; onde ainda não conseguimos, está marcado como **[a fazer]**.

> Regra que seguimos: se não sabemos explicar por que uma linha existe, ela não fica.

## Episódios

1. **Desenho da representação (Parte 2).** Pedimos ao modelo três opções para o rótulo de
   fronteira. Escolhemos "fronteira só entre instâncias" (max/min-filter no mapa de ids)
   em vez de "borda de todo objeto" porque é o pixel que a semântica não separa, e o
   watershed regrows o interior até o fundo de qualquer jeito. A distância normalizada
   por instância foi sugestão da IA; a justificativa (dois picos separados por um vale
   no contato) verificamos com o oráculo: watershed com rótulos perfeitos dá mAP 0,96,
   CC ingênuo dá 0,14 na mesma imagem.

2. **Métrica de instância (Parte 1).** A IA escreveu `iou_matrix` com `np.bincount` sobre
   pares de ids (em vez de dois loops) e as duas regras de matching. Nós conferimos o
   guloso à mão numa imagem de 3 objetos e testamos que pred = GT dá mAP 1,0. A escolha
   de AP_τ = TP/(TP+FP+FN) (DSB2018) foi nossa — sem score por instância, não há PR.

3. **PyTorch sem GPU.** O ambiente não tinha CUDA; o wheel do PyPI exige libs NVIDIA. A IA
   gerou *stubs* `.so` para `libcuda`/`libcudnn` para o `import torch` funcionar em CPU.
   Isso não afeta o código do PA (roda normal em qualquer máquina com torch instalado).

4. **Ablações instáveis (Parte 3).** Com 60 s de treino, seeds da mesma configuração
   davam mAP 0,007 e 0,227. Investigamos as curvas: a perda caía, mas o IoU da classe
   interior era 0 — a rede pequena "chamava tudo de fronteira" porque o α automático dava
   peso ~5× à classe minoritária. A IA propôs um teto (`--alpha-max 3`); nós escolhemos o
   valor e subimos o orçamento para 100 s. A instabilidade restante entre seeds é real e
   está reportada como desvio-padrão.

5. **Correção que não funcionou (Parte 5).** O diagnóstico inicial ("fronteira de 2 px é
   fina demais → fusões") sugeria treinar com 3 px. Piorou: 127 → 166 fusões. Testamos as
   hipóteses seguintes (limiar do interior, abertura dos marcadores, objetos sem
   interior no rótulo — só 3,3%): nenhuma reduziu as fusões, que ficam em ~130 em todas.
   Conclusão que defendemos na apresentação: as fusões restantes são contatos sem
   evidência fotométrica (mesma intensidade + blur + ruído), limite da rede neste
   orçamento, não do decodificador. Marcadores pela distância foram a mudança que ajudou
   (fantasmas 39 → 15, mAP +0,01).

6. **Mosaico (Parte 4).** A primeira versão colava 9 imagens do teste — as emendas entre
   fundos diferentes geravam objetos-fantasma e contaminavam a medição. Trocamos por uma
   cena sintética grande gerada direto. A correção B (costurar mapas e rodar o watershed
   uma vez) foi ideia nossa depois de a IA implementar a fusão por IoU.

## O que é nosso e o que é da IA

| | |
|---|---|
| Da IA (revisado por nós) | estrutura do pacote, versões iniciais de todos os módulos, scripts das partes, README |
| Nosso | escolha da trilha, definições de fronteira/AP/matching, orçamentos e hiperparâmetros das ablações, decisão de descartar o mosaico colado, interpretação de todos os resultados, apresentação |
| **[a fazer]** | rodar em DSB2018 (loader e split já existem, não testado com os dados reais); rodar as ablações com GPU e ≥ 3 seeds; Eixo 3 (código pronto, não rodado) |

Ferramentas: Claude (Anthropic) via app; nenhum código de repositório de terceiros foi
colado.
