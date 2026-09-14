# AI_LOG.md — como usamos IA neste assignment

**Resumo honesto:** o esqueleto deste repositório (estrutura de pacotes, primeiras versões
dos módulos, scripts de cada parte, README) foi gerado com o Claude (Anthropic) a partir
do PDF do PA, e depois rodado e depurado em conversa. A primeira rodada de experimentos
(Parte 0 e versões sintéticas das Partes 1–6) foi executada num ambiente de 1 núcleo de CPU.
Depois, com o Claude Code rodando no nosso Mac (Apple M4 Pro), conferimos o repositório
contra o PDF, e todas as Partes 1–6 foram refeitas no DSB2018 na GPU (episódios 7–9). Tudo o que está aqui nós lemos, entendemos e conseguimos
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

7. **Checagem contra o enunciado.** Pedimos ao Claude Code para conferir item a item o PDF
   contra o repositório. O achado principal: tudo tinha rodado só no sintético, e o PDF
   trata o sintético como teste unitário ("antes de tocar em dados reais"). O loader do
   DSB2018 existia, mas nunca tinha sido executado. Baixamos o `stage1_train` e conferimos o
   split por modalidade (546 fluorescência / 108 histologia / 16 brightfield → 469/100/101).

8. **Três bugs que só apareceram na GPU (MPS).** (a) a perda não era movida para o device
   (o buffer α ficava na CPU → erro); (b) `torch.bincount` no MPS cai para a CPU e custava
   0,36 s por passo — trocamos por uma comparação por classe e conferimos que dá o mesmo α;
   (c) o treino continuava 3× mais lento que o benchmark da rede sozinha. Medindo separado o
   tempo de dados e de GPU, a causa era o batch **não contíguo** (o `transpose` HWC→CHW da
   imagem do DSB): as convoluções no MPS ficam ~3× mais lentas. Um `np.ascontiguousarray`
   levou a época de 36 s para 12 s. Também medimos que 4 treinos em paralelo na mesma GPU
   ficam 7,7× mais lentos cada, então os scripts rodam em sequência.

9. **Augmentação que não acontecia.** O `CachedDataset` materializava o recorte aleatório
   256×256 de cada imagem uma vez só; no DSB isso fixa o mesmo recorte em todas as épocas.
   O treino no DSB passou a não usar cache (o custo de gerar os rótulos é ~3 ms/amostra).

10. **O α que funcionava no sintético quebrou no DSB (vira a Parte 5).** O primeiro modelo
    final no DSB (focal balanceada, α automático) ficou com val mAP 0,18 e IoU de foreground de
    0,62 até em imagens de treino, ou seja, não era overfitting. A galeria mostrou p(fronteira)
    alta no núcleo inteiro e 914 fantasmas no teste. A conta explica: nos recortes do DSB a
    fronteira entre núcleos é 0,4% dos pixels (no sintético, 5%), e o α de frequência inversa
    normalizado vira 0,017 / 0,11 / 2,87 (fundo/interior/fronteira). Com α fixo 1/1/3, mesma rede
    e mesmo orçamento: val mAP 0,36, teste 0,234 → 0,502. As ablações no DSB usam esse α fixo.

11. **Orçamento com prazo.** Com o prazo apertado, as ablações no DSB usam um encoder menor
    (base 16, 15 épocas, ~2 min por treino) que o modelo final (base 32, 30 épocas, ~6 min).
    As conclusões da Parte 3 valem para esse regime curto, e isso está dito no RESULTS.

## O que é nosso e o que é da IA

| | |
|---|---|
| Da IA (revisado por nós) | estrutura do pacote, versões iniciais de todos os módulos, scripts das partes, README |
| Nosso | escolha da trilha, definições de fronteira/AP/matching, orçamentos e hiperparâmetros das ablações, decisão de descartar o mosaico colado, interpretação de todos os resultados, apresentação |
| **[a fazer]** | ablações com ≥ 3 seeds e mais épocas; Eixo 3 (código pronto, não rodado) |

Ferramentas: Claude (Anthropic) via app e Claude Code (terminal, no nosso Mac); nenhum
código de repositório de terceiros foi colado.
