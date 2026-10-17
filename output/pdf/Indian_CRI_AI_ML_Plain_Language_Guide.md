# Indian software and AI/ML patent guidelines

## Strict summary and plain-language study guide

Based on the supplied 67-page **Guidelines for Examination of Computer Related Inventions (CRIs), 2025**, issued by the Office of the Controller General of Patents, Designs and Trade Marks. Prepared 14 October 2026.

**How to use the citations:** Every `Source p.` or `Source pp.` reference points to the original supplied PDF, not to the pages of this guide. Its printed page numbers match the PDF viewer's page numbers. Section numbers identify the original provision. The cover is p. 1 and contents are p. 2.

**Scope:** This is a detailed paraphrase of this particular document. Rules, definitions, court reasoning, tables, flowcharts and all 20 numbered examples are covered. Repeated passages are consolidated. Examples labelled **Document example** come from the PDF; examples labelled **Illustration** are added explanations, not Patent Office rulings. The document's suggested outcomes are kept conditional where it uses conditional language.

**Authority and limits:** The guidelines guide examination; they do not make law. The Patents Act and Rules prevail if there is a conflict, and court interpretations and the guidelines can change. The PDF refers to Annexures I and II, but neither annexure is included in its 67 pages. This guide cannot summarize those absent materials. It does not assess any particular invention or replace a prior-art search. [Source pp. 5, 10, 22; §§1.4-1.5, 3.5, 4.4.2]

## 1. The essential rules to remember

- **A useful app is not automatically a patentable invention.** A claim must concern an eligible new product or process, involve an inventive step, be usable in industry, and be explained clearly enough to reproduce. A business benefit or an AI label alone does not satisfy these requirements. [Source pp. 9, 16-23, 49-50; §§3.1, 4, 5.1-5.2]
- **Section 3(k) has four separate exclusions:** mathematical methods, business methods, algorithms, and computer programs per se. Passing one test does not establish that the entire invention is patentable. [Source pp. 9-10, 23-35; §§3.2-3.4, 4.5]
- **“Per se” means the program considered by itself.** The document allows room for implemented software inventions that solve a technical problem through technical means and produce a technical effect beyond the normal incidental effect of running a program. [Source pp. 8, 34-35; §§2.11, 4.5.4]
- **New hardware is not compulsory.** Software on existing general-purpose hardware can qualify for consideration when the implemented invention produces the required technical contribution. Merely listing a processor and memory does not create that contribution. [Source pp. 13-15, 24-25, 35-37; §§3.5.5-3.5.9, 4.5.1.2, 4.5.4]
- **The actual invention matters more than claim wording.** Calling excluded subject matter a “system,” “method,” “module” or “computer-readable medium” cannot rescue it. Supported method and system claims can both be included, even as independent claims. [Source pp. 22-23; §§4.4.1-4.5]
- **A business method remains excluded even when automated.** The PDF distinguishes a commercial strategy from a technical invention used in a commercial setting. A new pricing rule and a new secure transmission mechanism for a bank are different kinds of contribution. [Source pp. 11-12, 26-31; §§3.5.3, 4.5.2]
- **Abstract math and algorithm steps remain excluded.** The document distinguishes these from enabled technical implementations in which the math or steps serve a larger technical process. [Source pp. 23-34; §§4.5.1, 4.5.3]
- **Explain how it works, not just what it promises.** Describe the components, connections, steps, implementation and best method known to you. A skilled reader must be able to work it without your help or undue experimentation. [Source pp. 20-23; §§4.4-4.4.3]
- **AI disclosure follows the actual inventive contribution.** If the innovation is the learning implementation, explain the architecture and training. If it is the data pipeline, explain that pipeline and why it produces the technical result. [Source pp. 53-61; §5.2.3]
- **Describe essential training-data traits.** The PDF says that describing defining characteristics is sufficient in most cases, rather than supplying a huge or sensitive dataset itself. This does not permit hiding characteristics essential to reproduce the claimed effect. [Source p. 59; §5.2.3, Scenario 6]
- **Human inventorship matters in the document's AI treatment.** It treats autonomous or very limited human-intervention AI-generated inventions as non-patentable because AI cannot be the person claiming true and first inventorship. AI-assisted human inventions are not categorically excluded. [Source p. 50; §5.2.1]
- **Examples are illustrations, not approvals.** A result described as outside one section 3(k) exclusion must still meet novelty, inventive step, industrial applicability, disclosure and other applicable exclusions. [Source pp. 16, 23, 36, 53; §§4, 4.5, 4.5.4.2, 5.2.3]

**Illustration:** “Our AI organizes documents better” states a benefit. An implemented storage/indexing design that explains its operations and demonstrates reduced memory or retrieval time has the kind of technical contribution the PDF asks examiners to investigate. Novelty and non-obviousness still require separate examination. [Derived illustration; source pp. 16-19, 35, 39-41]

## 2. What the document covers and why it exists

Computer-related inventions involve computers, networks or programmable equipment, with one or more features implemented wholly or partly through hardware or software. The guidelines aim to make Patent Office examinations consistent and clarify the section 3(k) exclusions. They also explain ordinary patentability requirements and relevant court decisions. [Source pp. 4-5; §§1.2-1.4]

The introduction discusses AI, ML, NLP and large language models; cloud and edge computing; IoT and smart cities; cybersecurity and AI threat detection; blockchain; homomorphic encryption and differential privacy; quantum computing; and 5G/6G, AR/VR and autonomous vehicles. These are the technological context, not blanket declarations that any invention in those fields is patentable. [Source pp. 3-4; §1.1]

**Illustration:** Moving an ordinary record-keeping app to the cloud changes its deployment. To support the kind of eligibility discussed here, identify the actual technical invention, such as a specifically implemented improvement in distributed-system consistency or fault tolerance. [Derived illustration; source pp. 35, 41-42]

## 3. The technical words translated

These are the PDF's 13 defined terms. Where the PDF relies on a statute, the explanation below paraphrases that definition. Its dictionary definitions are identified separately. Examples are explanatory illustrations. [Source pp. 5-8; §2]

- **Algorithm:** A set of rules or steps for solving a problem. The cited court treats devising it as an intellectual exercise; original expression may have copyright protection, which is different from patent protection. *Example:* instructions for sorting numbers. [Source pp. 5-6; §2.1]
- **Computer:** An electronic, magnetic, optical or other high-speed system that performs logical, arithmetic and memory functions, including connected input, output, processing, storage, software and communication facilities. *Example:* a server with its storage and network equipment. [Source p. 6; §2.2]
- **Computer network:** Computers interconnected through communication media and terminals; the connection need not stay continuously active. *Example:* devices that connect to a server intermittently over a wireless link. [Source p. 6; §2.3]
- **Computer program:** Instructions in words, code, schemes or another form, including machine-readable form, that cause a computer to perform a task or produce a result. *Example:* code that processes an image. [Source p. 6; §2.4]
- **Computer system:** A device or collection of devices with supporting input/output equipment, programs, instructions and data that performs computing functions. The quoted definition excludes the described non-programmable calculators. *Example:* a workstation with connected input and output devices. [Source p. 6; §2.5]
- **Data:** Formally represented facts, information, knowledge, concepts or instructions intended for, undergoing or already having undergone computer processing; it can be in internal memory or other media. *Example:* sensor readings stored for processing. [Source pp. 6-7; §2.6]
- **Firmware:** The PDF's dictionary definitions describe software or data stored so that it cannot be changed or lost, including on a chip. This is the definition used in this document, not a claim that all modern firmware is technically unchangeable. *Example:* embedded device-control instructions. [Source p. 7; §2.7]
- **Function:** Includes computing logic, control, arithmetic, deletion, storage, retrieval and communications. *Example:* storing a record or sending it to another device. [Source p. 7; §2.8]
- **Hardware:** The physical and electronic parts of a computer, rather than its instructions. *Example:* the processor or network card. [Source p. 7; §2.9]
- **Information:** Includes data, messages, text, images, sound, voice, code, programs, software, databases and the media described in the quoted statute. *Example:* a voice recording or database. [Source pp. 7-8; §2.10]
- **Per se:** Something considered by itself, without the additional surrounding facts. *Example:* program instructions viewed in isolation, rather than an implemented technical invention. [Source p. 8; §2.11]
- **Software:** Programs or instructions that operate and control a computer, using the PDF's dictionary meanings. *Example:* an operating system. [Source p. 8; §2.12]
- **Manual:** The Manual of Patent Office Practice and Procedure issued by CGPDTM, as amended, unless the context requires otherwise. *Example:* an examination procedure referred to by these guidelines. [Source p. 8; §2.13]

Other recurring language: **claim** = the boundaries of the protection requested; **prior art** = earlier relevant disclosures; **person skilled in the art / PSITA** = the competent technical reader used to assess the invention; **enablement** = enough teaching to work the invention without undue experimentation; **technical effect** = the technical result produced by the implemented solution, beyond merely running ordinary software. These are explanatory translations of the PDF's usage. [Source pp. 15-23, 35, 51-53]

### 3.1 A working glossary for the later examples

These are explanatory reading aids for technical terms used in the PDF, not additional statutory definitions or patent requirements. The citations locate where the document uses the relevant terminology.

- **AI / ML / DL:** Artificial intelligence is the broad field; machine learning learns behavior from data; deep learning uses multi-layer neural networks. A **model** is the trained or trainable mechanism that maps inputs to outputs. *Example:* an image enters and a crop-type prediction leaves. [Terminology used on source pp. 49-50, 56]
- **Architecture, layer, activation and parameters:** Architecture is how the model is organized; layers are successive processing stages; activations are rules applied within those stages; parameters are the adjustable values learned or used by the model. ReLU and GELU are named activation choices. *Example:* explaining a model means describing the stages and their connections, rather than just calling it “deep learning.” [Source pp. 54, 58, 60]
- **ANN / CNN / transformer:** An artificial neural network connects computational units. A convolutional neural network processes local patterns, such as image features. A transformer uses attention to relate relevant parts of its input. An **encoder** produces a representation; a **decoder** turns a representation into an output. *Example:* the prescription system extracts image features, then generates text. [Source pp. 49-50, 53-54]
- **RNN / LSTM / time series:** Recurrent networks process sequences; long short-term memory is a recurrent-network design that retains useful sequence information. A time series is measurements ordered in time. *Example:* turbine readings over a time window help predict a coming failure. [Source pp. 49, 55]
- **Other named models:** Decision trees use branching decisions; random forests combine trees; Bayesian networks describe probabilistic relationships; expectation-maximization can fit models used in clustering; t-SNE reduces many-dimensional data to a lower-dimensional representation. Spiking networks use spike-like signals; GANs pair a generator with a competing discriminator; a perceptron network is another neural-network design. *Example:* naming one of these families tells the reader the general method, not your specific invention. [Source pp. 49-50, 58]
- **Training data, labels, annotations and features:** Training data supplies examples; labels/annotations mark the expected meaning or answer; features are the inputs the model uses. **SCADA** refers to supervisory control and data acquisition; its logs can supply industrial measurements. *Example:* label sensor sequences according to whether a machine later failed, then explain the sensors and sampling. [Source pp. 54-55, 58-59]
- **Training, loss, optimizer and hyperparameters:** Training adjusts the model; loss measures the error/objective being optimized; the optimizer decides how to adjust learned values. Hyperparameters are choices controlling that process, such as learning rate. Adam is a named optimizer, and cross-entropy is a named loss. *Example:* disclose how the turbine model learned from labelled failure records. [Source pp. 54-55, 60]
- **Epochs, batch size, gradient descent and backpropagation:** An epoch is a pass through the training examples; batch size is how many examples are processed together. Gradient descent changes values using the direction suggested by the objective; backpropagation calculates how model values contributed to error. *Example:* describing the training procedure includes these settings where they are needed to reproduce it. [Source pp. 50, 54, 60]
- **Regularization, dropout, normalization and augmentation:** These are training/data techniques: constrain the model to reduce overfitting; omit some units during training; standardize values; or create transformed training examples. *Example:* explain dropout rates and relevant normalization in the LSTM implementation. [Source pp. 54-55, 60]
- **Attention, embeddings and temporal processing:** Attention weights relevant input information; embeddings are numerical representations; temporal processing tracks patterns over time. *Example:* seasonal satellite features are represented and processed across time, rather than only as isolated images. [Source pp. 54, 59-60]
- **Preprocessing, segmentation and NDVI:** Preprocessing prepares raw data; segmentation divides an image into relevant regions; the Normalized Difference Vegetation Index is a calculated indicator of vegetation. Atmospheric correction adjusts satellite observations for atmospheric effects. *Example:* document how the raw satellite image becomes the crop classifier's prepared input. [Source p. 56]
- **Validation, test set, cross-validation and benchmark:** Evaluate performance on appropriate data/workloads; a test set supplies evaluation examples; cross-validation rotates held-out portions; a benchmark provides a comparison. **Generalization** means working on suitable unseen inputs, rather than merely remembered training examples. *Example:* compare the elderly-face system against the stated baseline, not just its training performance. [Source pp. 54-55, 58-60]
- **Accuracy, recall, false positive and false negative:** Accuracy counts correct predictions overall; recall asks how many actual positive cases were found; a false positive incorrectly flags a case; a false negative misses one. **Robustness** means maintaining the intended behavior under the relevant variations. *Example:* report missed faults and false alarms, not just “high accuracy.” [Source pp. 53-55, 59]
- **Reinforcement learning, state, action, reward and DQN:** An agent learns from interaction and feedback. State is what it observes/represents; action is what it can do; reward is the feedback objective. A deep Q-network uses a neural network to learn action values. The PDF also uses “advantage” language for the traffic example's feedback calculation. *Example:* queue length is observed, light timing is changed, and waiting-time reduction guides learning. [Source p. 57]
- **Kernel, scheduler, CPU core and cache:** The kernel is the operating system's central control layer; a scheduler assigns execution; a core performs processing; a cache holds quickly accessible data. A kernel hook/system-call interface is an integration point. *Example:* show where the learned scheduling decision changes which core runs a task. [Source p. 58]
- **Latency, throughput, bandwidth and I/O:** Latency is delay; throughput is work/data completed per unit time; bandwidth is communication capacity; input/output is movement of data into or out of a device. *Example:* faster retrieval or lower communication delay can be the technical result that needs explanation. [Source pp. 37-40, 58]
- **NIC, HSM, AES-GCM and IV:** A network interface controller connects a device to the network; a hardware security module performs protected security operations; AES-GCM is the named authenticated-encryption mode; an initialization vector is an input used with encryption. **Entropy** concerns unpredictability in random-source data. *Example:* the secure-transmission example assigns generation, encryption and packet delivery to specific components. [Source pp. 33-34]
- **NFC, IoT, protocols and actuators:** Near-field communication is short-range contactless communication; the Internet of Things connects devices/sensors; a protocol defines communication rules; an actuator carries out control, such as moving a joint or changing a light. *Example:* show how a sensor reading reaches the controller and produces an actuator command. [Source pp. 25-26, 30-31, 40, 57]
- **Quantum terminology:** A qubit is a quantum information unit; gates control its state; superposition and entanglement describe quantum behavior. Fidelity concerns closeness to intended operation; decoherence concerns loss of quantum behavior through interaction with the environment. Cryogenic cooling means extremely low-temperature operation. *Example:* disclose the actual qubit-control pulses, cooling and feedback needed for the claimed quantum system. [Source pp. 61-64]
- **Blockchain, consensus, finality and fault tolerance:** The ledger is shared among nodes; consensus is how they agree on accepted records; finality concerns confirmed records; fault tolerance concerns continued operation despite specified failures. PoA uses authorized validators; PBFT is Practical Byzantine Fault Tolerance. *Example:* explain who validates records and the limits of the stated hybrid mechanism. [Source pp. 64-66]
- **Hash, signature, Merkle tree and anchoring:** A hash is a derived digest; a digital signature supports verification of signing; a Merkle tree combines hashes in a tree to support integrity checks. Anchoring links an external record to an on-chain integrity reference. *Example:* store bulk sensor data off-chain while putting the specified integrity reference on-chain. [Source pp. 64-66]
- **Smart contract, oracle, on/off-chain, IPFS and MQTT:** A smart contract is executable ledger-related logic; an oracle supplies external information; on-chain data is on the ledger and off-chain data is elsewhere. IPFS is the named distributed storage approach; MQTT is the named messaging protocol. *Example:* an oracle provides traffic information while MQTT carries sensor messages and the contract applies the disclosed control logic. [Source pp. 64-66]

## 4. The law and court reasoning the PDF relies on

### 4.1 Basic statutory requirements and other exclusions

Section 2(1)(j) defines an invention as a new product or process involving an inventive step and capable of industrial application. Section 2(1)(ja) requires technical advance over existing knowledge, economic significance, or both, **and** non-obviousness to a skilled person. Section 2(1)(ac) requires that it can be made or used in industry. These requirements are cumulative; commercial value does not remove section 3 exclusions. [Source p. 9; §3.1; source p. 23; §4.5]

The PDF quotes additional section 3 exclusions: **3(l)** covers literary, dramatic, musical, artistic and other aesthetic creations, including films and television productions; **3(m)** covers mere schemes, rules, mental acts and methods of playing games; **3(n)** covers presentation of information; **3(o)** covers integrated-circuit topography. Its detailed examination tests focus on 3(k); it does not explain every exclusion in the Act. [Source p. 9; §3.2]

**Illustration:** A new chess rule, a dashboard that only changes presentation, or the artistic output of a music generator can face the quoted exclusions even when produced through software. Do not treat section 3(k) as the only possible issue. [Derived illustration; source pp. 9, 42-43]

The Joint Parliamentary Committee's explanation is that “per se” was added to avoid rejecting otherwise qualifying inventions merely because a computer program had other things developed on or around it; programs as such were not intended to receive patents. The four limbs remain distinct, and the modifier is attached to computer programs, not to business methods. [Source pp. 9-12; §§3.3-3.5.3]

### 4.2 Every principal CRI decision discussed in section 3.5

- **Ferid Allani v Union of India, 12 December 2019:** Use technical effect and technical contribution to examine computer-program-based inventions. A program basis alone need not defeat eligibility; ordinary patent requirements still apply. *Plain example:* investigate the actual technical result of implemented software. [Source p. 11; §3.5.1]
- **Microsoft, Delhi High Court, 15 May 2023:** The quoted authentication solution used cookies and two-factor authentication at the network level, beyond ordinary interface design. *Plain example:* improving how network authentication works differs from rearranging the login screen. [Source p. 11; §3.5.2]
- **OpenTV, Delhi High Court, 11 May 2023:** The business-method exclusion is absolute for an invention whose substance is a business or administrative solution. UK/European reasoning about technical effects of business methods cannot simply be imported because their statutory wording differs. *Plain example:* running a new fee strategy on powerful servers does not cure its commercial substance. [Source pp. 11-12; §3.5.3]
- **Microsoft, Madras High Court, 3 July 2024 - mathematics:** A formula's presence does not automatically make a claim a mathematical method; abstract calculation and equation-solving remain excluded. The PDF mentions Brent's root-finding and Adams' differential-equation methods as illustrations of excluded mathematical methods. *Plain example:* distinguish claiming the calculation itself from using it in an implemented technical process. [Source pp. 12-13; §3.5.4]
- **Raytheon, Delhi High Court, 15 September 2023:** Requiring novel hardware has no basis in law; examine the claimed technical contribution/effect. *Plain example:* an eligible software implementation may use an existing processor. [Source p. 13; §3.5.5]
- **Microsoft, Delhi High Court, 16 April 2024:** A general-purpose-computer implementation must directly produce a specific, credible technical enhancement beyond ordinary computing. The quoted media-compression implementation integrated transforms, reversible overlap operations, a processor and a storage buffer, improving compression and reducing storage requirements. *Plain example:* explain the implemented mechanism that makes the computer perform better. [Source pp. 13-14; §3.5.6]
- **Microsoft, Madras High Court, 3 July 2024 - technical solution:** The quoted reasoning concerns improved system functioning/efficacy or a technical solution to a technical problem, rather than impact limited to a particular application or dataset. Read this alongside the PDF's later data-specific AI examples, not as a blanket ban on applications or dataset innovations. *Plain example:* show the technical consequence of the claimed implementation. [Source p. 14; §3.5.7; source pp. 59-61; §5.2.3]
- **Ab Initio, Delhi High Court, 30 July 2024:** An inventive input/process that enables faster, more efficient processor computation can be a technical effect without physical hardware changes; the effect goes beyond the usual interface. *Plain example:* a specifically implemented input-processing design improves computation. [Source pp. 14-15; §3.5.8]
- **BlackBerry, Delhi High Court, 30 August 2024:** Bare flow-control policies or instructions are not enough. Where algorithms are implemented in software and result in technical effect/contribution, software eligibility analysis can apply. The inventive feature must be the implementation, not the isolated algorithm. *Plain example:* disclose the implementation that improves technical operation, rather than only listing data-flow rules. [Source p. 15; §3.5.9]

**How these fit together:** “No novel hardware required” is not “no technical implementation required.” The document requires a claimed, enabled technical solution and examines the claim as a whole. Its court extracts are interpretive examples, not a substitute for all statutory tests. [Source pp. 13-15, 22-23, 35]

## 5. The ordinary patent tests still apply

### 5.1 Novelty: was this already disclosed?

The claimed subject matter must not have been disclosed before the filing date or applicable priority date. The PDF uses the Ericsson v Lava decision of 28 March 2024 and its seven-step “Seven Stambhas” approach. Keep novelty separate from inventive step. [Source pp. 16-17; §4.1]

1. Understand the claims and the boundaries they define.
2. Identify relevant earlier disclosures: public disclosures, publications, patents and patent applications.
3. Compare the relevant technical details, documenting similarities and differences.
4. Check both explicit disclosure and implicit disclosure of the same invention.
5. Identify material differences across the entire claimed scope.
6. Assess the claimed combination as a whole, not merely whether its individual elements were known.
7. Record the conclusion with the specific prior-art passages and reasons supporting it. [Source pp. 16-17; §4.1]

The PDF also refers to the Act, Rules and Manual chapter 09.03.02 for novelty procedure. [Source p. 17; §4.1]

**Illustration:** Known sensors, known models and known processors do not alone answer whether your claimed combination is new. Compare the actual combination against relevant earlier disclosures, then separately ask whether the difference was obvious. [Derived illustration; source pp. 16-19]

### 5.2 Inventive step: would a skilled engineer find it obvious?

Novelty alone is insufficient. The invention as a whole must satisfy the statutory technical-advance/economic-significance requirement and non-obviousness. The PDF relies on **Biswanath Prasad Radhey Shyam v Hindustan Metal Industries**, **F. Hoffmann-La Roche v Cipla** and **Ericsson v Lava**. It emphasizes an objective assessment at the priority date without knowledge of the claimed invention. Do not modify the Indian test by importing additional imagined qualities of the skilled person from foreign decisions. [Source pp. 17-19; §4.2]

1. Identify the relevant skilled person: a competent craftsman or engineer.
2. Establish that person's common general knowledge at the priority date.
3. Identify the claim's inventive concept, or interpret the claim to determine it.
4. Identify differences between the prior art/state of the art and that concept.
5. Without hindsight, decide whether those differences were obvious or required inventive ingenuity. [Source p. 19; §4.2]

**Illustration:** Replacing a familiar rule with a familiar model might be an ordinary engineering choice. The question is whether the actual claimed implementation would have been obvious using the knowledge available then, not whether it took effort to build. [Derived illustration; source pp. 18-19]

### 5.3 Industrial applicability: can it actually be used?

The invention must be capable of being made or used in industry. It must be workable and useful; the description and common general knowledge must make a practical use and concrete benefit apparent. Speculation about a future use or a vague objective is insufficient. [Source p. 20; §4.3]

**Illustration:** A reproducible system that controls a factory robot has a concrete use. A claim that an undefined AI “may one day optimize any industry” does not disclose the workable use this document asks for. [Derived illustration; source pp. 20, 25-26, 49]

## 6. Disclosure and claim drafting: explain the invention completely

### 6.1 What the complete specification must contain

Section 10(4), as described here, requires a full and particular description of the invention, operation/use and method of performing it; the best method known to the applicant and entitled to protection; claims defining the requested scope; and an abstract providing technical information. Patent protection is exchanged for teaching the invention to the public. [Source pp. 20-21; §4.4]

The specification must be clear, correct, accurate and unambiguous. A skilled person must be able to work it without the applicant's assistance and without undue experimentation. Misleading statements and a description limited to functionality do not meet that standard. [Source p. 21; §4.4]

- **“What” for a system/device:** Explain every feature with appropriate drawings, the working relationship and connectivity of components, intermediate components and the intended output.
- **“What” for a method:** Explain the necessary sequence, flowcharts, implementation mechanism, intermediate steps and information needed to perform it and distinguish it from prior art.
- **“How”:** Explain the best implementation/use with suitable illustrations. The particularly claimed novel and inventive solution must be fully taught. [Source p. 21; §4.4]

**Illustration:** “A module detects faults” tells the reader a desired function. Describe the inputs, transformations, implemented component, integration and resulting fault signal sufficiently to teach the claimed solution. The PDF warns that a problem statement or imagined use case can be disguised as if it were already a solution. [Derived illustration; source pp. 21, 50-53]

### 6.2 Claims: clear boundaries supported by the description

Claims must satisfy unity of invention under section 10(5), be clear and succinct, and be fairly based on what the specification discloses. Their substantive meaning matters more than whether they are written as a method, system, device, program product or storage medium. [Source p. 22; §§4.4.1-4.4.2]

Both method and system claims may be allowed, including independent claims, when the description supports them. There is no rule in this PDF that only one of these claim forms can be granted. Conversely, neither form protects excluded subject matter merely through a change of wording. [Source p. 22; §4.4.2]

**Illustration:** Changing “a pricing rule” to “a system comprising a processor that applies the pricing rule” does not change its commercial substance. Describe and claim an actual technical invention if there is one. [Derived illustration; source pp. 22, 29-30]

### 6.3 “Means for” claims need disclosed structure

If a claim says “means for performing X,” the specification must identify the structure that performs X, using structural features and reference numerals to make it understandable. The structure may be specific hardware, firmware or a software module disclosed in the description and drawings. Such means are limited to what is disclosed; undisclosed structure cannot support the claim. [Source pp. 22-23; §4.4.3]

**Illustration:** “Means for converting digital to analog” needs the corresponding disclosed implementation, not an unexplained box bearing that label. [Source example translated; source p. 23]

## 7. The four section 3(k) exclusions, with every worked example

### 7.1 Mathematical methods

Pure calculation, equation formation, square/cube-root finding, abstract idea manipulation and purely mathematical problem-solving are excluded. A mathematical formula in a claim does not by itself cause exclusion: the PDF distinguishes computation as the claimed objective from computation as part of a larger technical process. Encoding, communication noise reduction and electronic encryption/decryption are mentioned as possible technical applications. [Source pp. 23-24; §4.5.1]

**The test:** Interpret the entire claim and its main objective. If the essence is abstract mathematical/statistical operations producing a numerical output, it falls under this exclusion. If calculation serves a broader technical objective rather than being the main aim, the document says it does not fall under this particular exclusion. [Source p. 24; §4.5.1.1]

**Document example 1 - dataset compatibility index:** A processor, memory and analysis module apply a predefined formula to two datasets and output a compatibility score. The entire purpose is the computation; no larger technical process is disclosed. **Outcome: excluded as a mathematical method.** Adding computing components does not change that. [Source pp. 24-25; §4.5.1.2]

**Document example 2 - robotic-arm control:** Target coordinates and optional orientation are converted through inverse kinematics, using the arm's geometry, joint limits and collision constraints. Calculated angles generate actuator commands, with sensor feedback for accurate positioning. The calculation serves physical control of the arm. **Outcome: outside the mathematical-method exclusion in this example; not a finding that every robotic-control claim is patentable.** [Source pp. 25-26; §4.5.1.2]

### 7.2 Business methods

The document describes business methods broadly as commercial/administrative activities involving goods, services, trade, finance and online buying or selling. If the core contribution is a financial scheme, marketing strategy, customer-management rule or administrative process, using technology does not rescue it. [Source pp. 26-28; §4.5.2]

Business words or context alone do not prove exclusion. The PDF specifically mentions enterprise, business rules, supply chain, order, sales, transactions, commerce and payment, as well as profile/relationship matching, event planning, credit, employee scheduling, customer feedback and relationship management. Investigate whether the actual contribution improves the underlying technical system through technical means. [Source pp. 26-27; §4.5.2]

**The test:** Find the core of the whole claim. A primarily commercial/administrative strategy is excluded. A technical improvement to infrastructure or a process, with business context only defining its setting, is treated differently. The flowchart additionally asks whether the technical character provides a technical solution to a technical problem through technical means. [Source p. 28; §4.5.2.1; source p. 45; §4.5.5]

**Additional document contrasts:** A bank's tiered fee calculation is a revenue/customer-management strategy; a new cryptographic transmission technique or technically optimized server data handling is a technical contribution used by a bank. A tiered loyalty-point and discount program remains a business strategy even if computers manage its memberships. [Source pp. 27-28; §4.5.2]

**Document example 3 - dynamic advertisement pricing:** A server uses bids, historical ad performance and business rules to set prices and select advertisements. Its core is revenue optimization, not improved technical infrastructure. **Outcome: excluded as a business method.** [Source pp. 29-30; §4.5.2.2]

**Document example 4 - NFC tap-and-pay POS:** The system configures an NFC reader, receives card data, uses a secure protocol for authentication/verification, requests authorization and updates the terminal. The PDF treats the contribution as improved technical payment infrastructure, with finance as the setting. **Outcome: outside the business-method exclusion in the example.** It does not declare ordinary NFC payments new or inventive. [Source pp. 30-31; §4.5.2.2]

### 7.3 Algorithms

Algorithms in all forms are excluded when the subject matter is the rule/procedure or finite instruction sequence itself. Listing sorting steps or abstract cryptographic steps without technical realization is insufficient. The document asks for enabling implementation details in a technical framework that produces a technical solution to a real-world problem. [Source p. 31; §4.5.3]

**The test:** Identify the series of steps after interpreting the whole claim. Are they abstract and missing implementation specifics/components? Then excluded. Are they enabled by specific technical components/details, explaining how implementation solves a technical problem? The document treats the implemented invention as outside this particular exclusion. This does not mean that adding any application name makes an algorithm patentable. [Source pp. 31-32; §4.5.3.1]

**Document example 5 - cryptographic key/pseudorandom-number generation:** Receive a seed, apply permutations and output numbers. The “permutation engine” lacks specifics: which permutations, how they run and how they produce a concrete technical solution. **Outcome: excluded as an algorithm.** [Source pp. 32-33; §4.5.3.2]

**Document example 6 - implemented secure transmission:** An HSM integrated into a NIC supplies entropy; a permutation unit generates 128-bit pseudorandom values into a volatile buffer; a cryptographic processor uses AES-GCM; an initialization vector is generated by a timestamp counter; a packetization engine packages and transmits the encrypted data. The PDF attributes improved security to this specific implemented interaction. **Outcome: outside the algorithm exclusion in the example.** This is a paraphrase of its eligibility illustration, not an independently validated cryptographic design. [Source pp. 33-34; §4.5.3.2]

### 7.4 Computer programs per se

Claims aimed only at instructions, routines/subroutines, program products, a storage medium carrying instructions, a database or memory holding instructions are excluded in the circumstances described here. The document leaves room for software inventions with a technical solution, technical means and technical effect beyond incidental execution. Novel hardware is unnecessary. [Source pp. 34-35; §4.5.4]

**The test:** (1) Interpret the claim as a whole and identify essential technical features. (2) Identify the core problem and proposed solution, and whether they are technical. (3) Decide whether that technical character creates a technical effect beyond an incidental effect. (4) An affirmative result places it outside the computer-program-per-se exclusion; otherwise it is excluded. [Source p. 35; §4.5.4.1]

**Document example 7 - recipe manager:** Store/search recipes, scale ingredients and generate shopping lists on a standard computer. The PDF sees personal information management and convenience, without improved computer/network operation or device control. **Outcome: excluded as a computer program per se.** [Source pp. 36-37; §4.5.4.2]

**Document example 8 - adaptive network optimizer:** Sensors monitor interference and traffic; software analyzes the readings; control commands dynamically change NIC transmission power, frequency or routing. The closed-loop implementation improves throughput, latency, robustness and resource use. **Outcome: outside the computer-program-per-se exclusion in the example.** [Source pp. 37-39; §4.5.4.2]

## 8. Every category in the two software eligibility tables

The tables are **non-exhaustive**, and assessment is case by case. The following preserves every listed category and its substance. These are indicators to investigate, not automatic patent approvals or a complete legal checklist. [Source pp. 36, 39-43; §4.5.4.2, Tables 1-2]

### 8.1 Table 1: technical contributions that may avoid program-per-se exclusion

- **Faster internal computation:** Innovative implemented algorithms or architecture interactions improve processor computation. *Example:* a disclosed input-processing mechanism makes calculation faster. [Source p. 39; Table 1]
- **Faster storage access:** Improved caching or disk input/output scheduling reduces access time. *Example:* explain a caching mechanism and how it cuts storage reads. [Source p. 39; Table 1]
- **Better memory use/management:** Reduce memory consumption or improve its efficiency. *Example:* an implemented memory layout uses less space for the same technical workload. [Source p. 39; Table 1]
- **Better database operation:** Implemented searching, indexing or retrieval improves overall system performance. *Example:* a technical indexing design reduces retrieval cost. [Source p. 39; Table 1]
- **Better compression/expansion:** Improve lossless/lossy compression rate or speed. *Example:* a disclosed codec mechanism reduces storage requirements. [Source p. 40; Table 1]
- **Better authentication, encryption and privacy:** Concrete security implementations, including network-traffic intrusion detection. *Example:* explain an authentication protocol that improves network security. [Source p. 40; Table 1]
- **Better resource allocation:** Technical CPU, bandwidth or cloud-resource management; the PDF mentions faster job scheduling in high-performance computing. *Example:* a disclosed scheduler reduces execution delays. [Source p. 40; Table 1]
- **Better error detection/correction:** Improve storage or transmission reliability. *Example:* an implemented correction mechanism handles corrupted transmitted data. [Source p. 40; Table 1]
- **Improved machine control:** Precise/adaptive operation of robots, industrial machines or 3D printers. *Example:* better arm positioning and movement. [Source p. 40; Table 1]
- **Improved signal handling:** Better reception, processing or transmission of radio/electromagnetic signals. *Example:* improved noise filtering or modulation/demodulation. [Source p. 40; Table 1]
- **Real-time monitoring/control:** Sensor processing improves equipment or environmental monitoring/control. *Example:* IoT or medical-sensor data changes how equipment operates. [Source p. 40; Table 1]
- **Advantageous embedded control:** Embedded code produces specific improved hardware behavior. *Example:* reducing an IoT device's power consumption based on usage. [Source p. 40; Table 1]
- **Autonomous vehicle/drone control:** Generate control signals from real-time sensor processing. *Example:* a specific sensor-processing pipeline directs a drone. [Source p. 40; Table 1]
- **Technical medical-image analysis:** Inventive implementations detect anomalies or improve image quality with a better technical outcome. *Example:* disclose the implemented image-processing improvement. [Source p. 41; Table 1]
- **Distributed/cloud systems:** Technically improve synchronization, consistency or fault tolerance. *Example:* an implemented synchronization mechanism reduces inconsistent replicas. [Source p. 41; Table 1]
- **Technical/physical simulation:** Accurately simulate systems, such as fluid dynamics or protein folding, to predict, control or assist design. *Example:* a technically implemented simulator improves a design process. [Source p. 41; Table 1]
- **Efficient ML training or inventive architectures:** Improve performance or reduce computation for a specific technical task. *Example:* an implemented training/architecture change reduces cost while performing that task. [Source p. 41; Table 1]
- **Technical signal processing:** Implement processing that solves a technical problem. *Example:* explain the processing mechanism that suppresses interference. [Source p. 41; Table 1]

### 8.2 Table 2: contributions that fall under the listed exclusion

- **Code in isolation:** Instructions, a program or program product considered on its own. *Example:* protection sought for the instruction set alone. [Source p. 41; Table 2]
- **Software defined only by its medium:** “Program on disk” or “medium storing instructions” without the required technical invention. *Example:* a medium containing ordinary data-processing code. [Source p. 41; Table 2]
- **Data structures detached from a technical result:** A format or arrangement without technical process improvement. *Examples:* a new file format without efficiency improvement; an organizational-chart hierarchy. [Source p. 42; Table 2]
- **Ordinary automation:** Manual record keeping or scheduling moved to a computer, without an additional technical gain beyond its inherent operation. *Examples:* ordinary bookkeeping/accounting automation; employee scheduling using standard database operations. [Source p. 42; Table 2]
- **Ordinary presentation:** Text/visual reports or dashboards without a technical data-handling/display solution. *Examples:* a new weather layout; business charts from standard tools. [Source p. 42; Table 2]
- **Game/mental-exercise rules:** Software implements rules or thinking strategies without the required technical invention. *Examples:* a new chess variant; a Sudoku-solving strategy; teaching methods. [Source p. 42; Table 2]
- **Human-reasoning imitation:** Decision logic without specific implementation yielding technical effect. *Examples:* if-then legal advice; symptom matching without technical innovation. [Source p. 42; Table 2]
- **Aesthetic/artistic focus:** The contribution is artistic creation. *Examples:* generating art/music; fashion layouts. The PDF associates such creations with copyright rather than patent protection. [Source p. 42; Table 2]
- **Rule-only medical diagnosis:** Correlation or decision trees mimicking a doctor's mental process, without technical processing/innovation. *Examples:* a symptom database lookup or diagnostic decision tree. [Source p. 43; Table 2]
- **Standard retrieval/lookup:** No technical improvement beyond normal querying. *Examples:* standard keyword matching; basic library catalog queries. The original row also repeats playlist language; its distinct substance is preserved here. [Source p. 43; Table 2]
- **Preference-only personalization:** Recommendations based only on preferences/history without technical system effect. *Examples:* genre-based playlists or basic user profiling. [Source p. 43; Table 2]

**Illustration:** Personalized content and faster internal retrieval are separate contributions. For a modern recommendation product, identify which one the claim actually covers and teach the specific technical implementation if that is the invention. [Derived illustration; source pp. 39, 43]

## 9. The four flowcharts in ordinary speech

The flowcharts repeat the four exclusion tests. They contain visual content that ordinary text extraction misses; they were checked as images. [Source pp. 43-48; §4.5.5]

1. **Mathematical method, p. 44:** Identify the real solution. Abstract math aimed only at producing an output is excluded. If it is instead part of a larger technical process contributing to a broader technical objective, it avoids this exclusion. The chart routes a failure to establish that broader process to exclusion.
2. **Business method, p. 45:** Identify the core. An administrative/commercial strategy is excluded. Otherwise check both technical character and a technical solution to a technical problem through technical means; a failure of either routes to exclusion.
3. **Algorithm, p. 46:** Identify the steps. Abstract steps missing implementation details are excluded. Enabled technical steps that produce a technical solution to a real-world problem avoid this exclusion; otherwise excluded.
4. **Computer program per se, pp. 47-48:** Interpret the whole claim, identify essential technical features and the core problem/solution, then assess technical character and effect. The chart's yes branch avoids this exclusion. The accompanying prose on p. 35 specifies that the effect must go beyond an incidental effect. [Source pp. 35, 44-48]

**Illustration:** For an ML fraud product, do not stop after naming a model. Work through what is actually claimed: a commercial decision rule, an abstract calculation, or an enabled technical security process. Each relevant exclusion must be considered. [Derived illustration; source pp. 23-35, 44-48]

## 10. AI, ML and deep learning: the general rules

AI/ML/DL, blockchain and quantum computing rely on mathematical models, algorithms and abstract principles. Their future potential or a published vision is not an implemented invention. The central question is whether the specification teaches a practical technical solution reproducible without undue experimentation. [Source p. 49; §5.1]

The PDF names ANNs, CNNs, RNNs, spiking networks, GANs and transformers; random forests, decision trees, Bayesian networks, expectation-maximization clustering and t-SNE; backpropagation, gradient descent and attention; and supervised, unsupervised and reinforcement learning. Their theoretical foundations are abstract. A tangible implemented application may qualify, subject to the patentability criteria. Listing a recognized architecture does not establish invention. [Source pp. 49-50; §5.2]

### 10.1 Inventorship

The PDF separates **AI-generated** inventions, created autonomously or with very limited human intervention, from **AI-assisted** inventions, where AI is a tool used in the inventive process. It says the former are not patentable because AI cannot be the “person” claiming true and first inventorship under section 6. It says the latter are not categorically excluded under 3(k), provided they meet patentability criteria and show technical effect through tangible inventive applications. This is a statement of the supplied document's treatment. [Source p. 50; §5.2.1]

**Illustration:** A human using AI tools to develop a technical implementation belongs to the second category described by the PDF; naming an autonomous AI as inventor raises the first-category problem. The guide does not decide who invented any particular project. [Derived illustration; source p. 50]

### 10.2 Why grand AI descriptions fail: the Motupalli example

In **Caleb Suresh Motupalli v Controller of Patents, 29 January 2025**, the quoted claims concerned persona extension/augmentation and integration of people, devices and environments. The specification named black-box modernization, distributed-object technology, middleware, object-oriented techniques and familiar devices, but did not teach how they actually produced the claimed integration. Collecting ideas from patent/non-patent literature did not supply the missing implementation. [Source pp. 50-52; §5.2.2]

The skilled reader may be one expert or a relevant team; here the court considered AI and allied software expertise. Such a reader must not need inventive effort or undue experiments to bridge the disclosure gap. The PDF describes failure of **10(4)(a)** enablement, **10(4)(b)** best-method disclosure and **10(5)** clear/succinct, fairly based claims. Best method is judged as a practical matter, and the description must enable the full claimed scope rather than include unworkable embodiments. [Source pp. 51-53; §5.2.2]

The court discussion cites **C Van Der Lely v Ruston's Engineering** for practical best-mode assessment, **General Tire v Firestone** for practical clarity of definition, and **Biogen v Medeva** for support/enablement over the full monopoly claimed. These are supporting authorities quoted within the PDF, not additional AI eligibility tests invented by this guide. [Source pp. 52-53; §5.2.2]

**Plain meaning:** A collection of impressive component names is not a recipe for building the claimed invention. Teach the inventive connection between those components and the result. [Source pp. 51-53]

## 11. All seven AI disclosure scenarios and examples 9-16

The scenarios are non-exhaustive. Their lists indicate what the disclosed invention needs in those situations; they do not set one universal accuracy threshold, training-set size or architecture requirement for every AI invention. Numerical X% and Y% values in several examples are hypothetical placeholders. [Source pp. 53-61; §5.2.3]

### 11.1 Scenario 1: the invention is the input-to-output transformation

Explain how the transformation occurs using the known processes and variables, sufficiently specifically for a skilled reader to reproduce it. Where test results/evidence validate output accuracy, include them, especially for precision-critical uses. The PDF asks for evidence of consistency, accuracy in expected varied conditions, robustness to small input changes and low critical-error rates. It illustrates these with repeatable image diagnoses, braking in different weather, fraud detection despite formatting changes, and quantified machinery-fault false positives/negatives. [Source p. 53; Scenario 1]

**Document example 9 - handwritten prescription to structured record:** A CNN such as ResNet extracts image features; a transformer decoder turns them into text and structured JSON with drug names, doses and instructions. The example describes one million pharmacist-labelled prescriptions and a separate 10,000-image test set. These are this hypothetical example's details, not mandatory dataset counts. [Source pp. 53-54; Example 9]

- Disclose network structure: layer types, depth and activation functions.
- Characterize handwriting variability, labels/annotation and the digital output format.
- Teach preprocessing implementation: binarization/cleaning, normalization, filtering, enhancement, noise removal and segmentation as applicable.
- Explain loss functions, optimizer settings and training epochs.
- Report validation/real-world results such as accuracy and recall showing generalization.
- The PDF identifies reduced transcription errors, prescription safety and improved processing of complex handwriting as possible effects supporting consideration outside 3(k). [Source p. 54; Example 9]

**Plain meaning:** A scanned image entering and a correct prescription record leaving are not enough. Teach the transformation and provide the stated evidence for the claimed reliable behavior. [Source pp. 53-54]

### 11.2 Scenario 2: the invention uses an already-trained model

Identify the training data and its distinctive characteristics, explain why they fit the technical problem, describe the model and training methodology, and show predictable/reliable technical results. The disclosure includes relevant hyperparameters, optimization, training duration and preprocessing, not merely the statement “use a trained model.” [Source pp. 54-55; Scenario 2]

**Document example 10 - machine-failure prediction:** Three stacked LSTM layers analyze turbine sensor time series, with dropout/batch normalization and an Adam/binary-cross-entropy training approach. The example refers to a public wind-turbine SCADA dataset and five-fold validation. [Source p. 55; Example 10]

- Identify dataset, sensor types, sampling frequency, collection duration and labelling.
- Explain dataset-to-task correlation and processing of time-series features; describe testing of stationarity, seasonality and trends if performed.
- Specify LSTM layers, input time window and dropout rates.
- Explain loss, learning rate and regularization.
- Give performance results showing predictable mechanical-failure prediction.
- The PDF identifies early anomaly detection, fewer outages and improved wind-farm operation as possible effects supporting consideration outside 3(k). [Source p. 55; Example 10]

**Plain meaning:** Knowing the model's name does not tell another engineer which signals it uses, how they were sampled, or how the trained system reliably predicts failure. [Source p. 55]

### 11.3 Scenario 3: preprocessing is central

Disclose every material preprocessing step/function and its connection to the final model. Missing the connection between raw inputs and prepared learning data can fail enablement. [Source p. 56; Scenario 3]

**Document example 11 - satellite crop classification:** Atmospheric correction using Sen2Cor, NDVI calculation, segmentation through vegetation indices/texture filters and an EfficientNet classifier turn raw satellite images into crop classifications. The example uses X% accuracy and Y% improvement as placeholders. [Source p. 56; Example 11]

- Explain the step-by-step pipeline, named correction algorithm, NDVI calculation and segmentation logic.
- Explain why preprocessing improves results over raw images.
- Disclose implemented CNN parameters and their relationships.
- Characterize the relevant training data and labels.
- Supply a comparison demonstrating the material preprocessing effect.
- The PDF identifies improved large-scale crop classification and agricultural monitoring/planning as possible effects supporting consideration outside 3(k). [Source p. 56; Example 11]

**Plain meaning:** If the invention is how you prepare the images, “we clean the data” does not teach it. Show the transformations and their contribution. [Source p. 56]

### 11.4 Scenario 4: reinforcement learning

Explain the agent's observation and influence on its environment, states, available actions and rewards/feedback values. Omitting essential details that a skilled reader cannot deduce can make the disclosure non-enabled. [Source pp. 56-57; Scenario 4]

**Document example 12 - Adaptive Urban Traffic Optimizer (AUTO):** A DQN with CNN processing of camera feeds represents traffic density/queues and changes light timings. A centralized agent communicates with signals through IoT protocols, trains in simulations and transfers to real deployment. [Source p. 57; Example 12]

- Describe DQN architecture, inputs, state representation, discrete/continuous action space and the precise advantage/feedback calculation linked to waiting-time reduction.
- Identify camera/detector sensors, traffic-light/signage actuators and the communication/control interactions.
- Explain simulation setup, simulator and scale, pretrained model, fine-tuning and source-domain characteristics.
- The PDF identifies reduced waiting, congestion, fuel use and CO2 emissions, and scaling across urban settings, as possible effects supporting consideration outside 3(k). [Source pp. 57-58; Example 12]

**Plain meaning:** Define exactly what the agent sees, can change and learns from, and how its decision reaches the actual traffic lights. [Source p. 57]

### 11.5 Scenario 5: AI improves the computer itself

Explain interaction with hardware/system architecture and how internal storage, scheduling or processing metrics improve. [Source p. 58; Scenario 5]

**Document example 13 - neural OS scheduling:** A lightweight, two-layer perceptron uses thread priority, cache hits and CPU utilization to select a CPU core. It replaces a default heuristic in the Linux Completely Fair Scheduler; X% latency and Y% throughput changes are illustrative placeholders. [Source p. 58; Example 13]

- Define input features and why they represent the relevant system state.
- Describe layers, activations and training.
- Explain integration at system-call/kernel-hook level and CPU/OS compatibility.
- Describe training-data sources and preprocessing where applicable.
- Validate latency/throughput effects.
- The PDF identifies better OS scheduling and CPU-resource use as possible effects supporting consideration outside 3(k). [Source p. 58; Example 13]

**Plain meaning:** Explain where the model enters the scheduler and how it changes execution, then support the performance claim. [Source p. 58]

### 11.6 Scenario 6: special training-data traits are the inventive basis

Disclose essential dataset traits unless a skilled reader can identify them without undue experimentation. Include relevant domain, diversity, precise labelling and unique characteristics. In most cases, the PDF says the defining characteristics suffice rather than the actual voluminous/sensitive dataset. [Source p. 59; Scenario 6]

**Document example 14 - elderly-person recognition:** A dataset of people aged 65-90 covers aging features, occlusions, lighting, poses and accessories. A modified FaceNet with age-aware loss is compared with a general-dataset baseline. X and Y are placeholders, with X greater than Y. [Source p. 59; Example 14]

- Describe the critical age distribution, facial characteristics and occlusions.
- Explain why general face datasets do not suffice and how selected traits change performance.
- Describe architecture changes and age-weighted loss.
- Quantify the improvement over baselines attributable to dataset selection.
- The PDF identifies reliable elderly identification, reduced false negatives and support for safety/health monitoring as possible effects supporting consideration outside 3(k). [Source p. 59; Example 14]

**Plain meaning:** “Our private dataset works better” hides the claimed teaching if its essential traits are unexplained. Describing the needed traits can differ from publishing every individual record. [Source p. 59]

### 11.7 Scenario 7: identify whether the invention is learning implementation or data engineering

If the inventive contribution is a new learning implementation, fully explain its unique elements, mathematical basis and operational steps. If the contribution is innovative creation, curation or use of data that makes a known algorithm achieve the result, focus disclosure on that data process. [Source p. 60; Scenario 7]

**Document example 15 - 7(a), new crop-learning implementation:** A modified EfficientNet adds attention such as squeeze-and-excitation/spatial attention, temporal convolution or transformers for seasonal patterns, and improved losses/training schedules. [Source p. 60; Example 15]

- Disclose layer-by-layer layout, input dimensions, output structure, parameter counts and functions such as ReLU/GELU.
- Specify loss, optimizer, batch size, epochs and augmentation.
- Benchmark against known datasets. [Source p. 60; Example 15]

**Document example 16 - 7(b), data creation and curation:** An automated pipeline harmonizes multi-season, multi-source satellite data through cross-sensor calibration, temporal stacking, gap filling and AI-assisted label validation/conflict resolution. Here the pipeline, not the subsequent classifier, is the claimed contribution. [Source p. 61; Example 16]

- Explain the sequence of data operations and deployment/use context; a generic supporting classifier can be described more lightly where it is not the inventive core.
- Explain why the particular steps are innovative and necessary for the required data quality.
- Show validation, for example against models using conventionally prepared data.
- For 7(a) and 7(b), the PDF identifies technically improved crop-feature extraction across seasons, accuracy/efficiency and better agricultural monitoring as possible effects supporting consideration outside 3(k). [Source p. 61; Examples 15-16]

**Plain meaning:** Put the detailed teaching where the invention actually is. A new learning implementation and a new dataset-production process need different explanations. Neither is protected simply because it uses ML. [Source pp. 60-61]

## 12. Quantum computing: the remaining modern-computing rules

The PDF discusses superposition, entanglement and tunnelling; sensing, quantum-key distribution, simulation, quantum internet and materials/devices; qubits, gates, chips, processors, cryogenic/shielding systems and amplifiers; and compilers, decoders, simulators, emulators, circuit visualization and error correction. Listed error-correction examples include Steane, Bacon-Shor, 3D color and surface codes. These identify the field, not automatic eligibility. [Source pp. 61-62; §5.3]

Abstract quantum principles/formulas are not patentable on that basis. A practical implementation with a concrete technical improvement may qualify. Although the text mentions logistics optimization as a possible application, it must still be read alongside the business-method and other tests; there is no blanket exception for using quantum computing. [Source pp. 26-28, 49, 62; §5.3]

### 12.1 Document example 17 - hybrid quantum/classical processor

A transmon-qubit QPU using niobium-titanium/sapphire materials works with an ML classical controller, a compiler translating high-level code to qubit-specific pulses, cooling/shielding at 10-15 millikelvin, and synchronization of gates with classical processing. [Source p. 62; Example 17]

- Disclose qubit materials, dimensions and design.
- Explain controller architecture, model parameters, training-data types and feedback protocols.
- Show compiler-to-pulse flow, timing and qubit-specific error margins.
- Specify cryogenic and electromagnetic shielding requirements.
- Provide block diagrams linking the QPU, controller, compiler and synchronization modules.
- The PDF identifies adaptive gate adjustment, higher fidelity and reduced incoherence as possible effects supporting consideration outside 3(k). [Source p. 63; Example 17]

**Plain meaning:** “Use ML to make a quantum computer better” is a goal. Teach the control loop, physical requirements and pulse-level integration. [Source pp. 62-63]

### 12.2 Document example 18 - photonic quantum chip

A CMOS-compatible optical chip includes waveguides, beam splitters, phase shifters, quantum-dot photon sources, micro-heater tuning, superconducting nanowire photon detectors, delay lines and feedback for linear-optical quantum protocols. [Source p. 63; Example 18]

- Disclose layout, waveguide geometry/materials and fabrication.
- Explain control logic for single- and two-qubit gates.
- Provide insertion-loss, extinction-ratio and phase-tuning-response data.
- Give sample LOQC execution, timing diagrams and expected output distributions.
- The PDF identifies high-speed/low-noise operation, gate fidelity, scalability, thermal-decoherence immunity, precise control and integrated miniaturization as possible effects supporting consideration outside 3(k). [Source pp. 63-64; Example 18]

**Plain meaning:** Explain the buildable optical circuit and its measured behavior, not just the quantum principle. [Source pp. 63-64]

## 13. Blockchain: technical implementation and disclosure

The PDF describes a peer-to-peer, distributed, append-only ledger maintained through node consensus rather than one central server. It discusses hashes, signatures, non-repudiation, encryption, smart contracts and distributed networks, with applications in banking, security, supply chains, healthcare and governance. Abstract blockchain ideas are not enough; practical implemented technical effects are central. [Source p. 64; §5.4]

Disclose cryptographic techniques, data structures, block layout/linkages, consensus and hardware/network interaction sufficiently to reproduce the invention. Define the ledger, smart contracts, cryptographic processes and network configuration. The PDF lists proof of stake, delegated proof of stake, work, burn, deposit, importance and activity, multisignature and PBFT as consensus-related examples. [Source pp. 64-65; §5.4.1]

The PDF permits high-level smart-contract functionality while keeping sensitive implementation details undisclosed **if the technical effect does not lie in the smart contract**. This is conditional: it does not authorize concealing implementation essential to enable the actual claimed invention. [Source pp. 64-65; §5.4.1; source pp. 20-23; §4.4]

### 13.1 Document example 19 - blockchain rental and access

A smart contract defines rental terms/access conditions, verifies a tenant's identity/payment request, produces an access token, activates a physical or virtual asset through an access module, and records timestamps/payment events. [Source p. 65; Example 19]

- Explain contract logic, parameters and automatic trigger conditions.
- Disclose protocol/interface integration with IoT hardware.
- Specify the immutable records, such as hashes, timestamps and user IDs.
- Explain fallback behavior: failed execution/payment, manual override, external oracle, expired contracts and disputed access.
- The PDF identifies decentralized access/payment enforcement, reduced fraud/latency and secure conditional IoT access as possible effects supporting consideration outside 3(k). [Source p. 65; Example 19]

**Plain meaning:** Teach how verified events control the asset and what happens when they fail. The document describes possible technical aspects; it does not guarantee that a rental-business claim escapes the business-method exclusion. [Source pp. 26-28, 65]

### 13.2 Document example 20 - supply-chain provenance

A private blockchain combines PoA and PBFT, smart contracts for origin/compliance/routing, oracles for external route information, and off-chain sensor storage anchored on-chain using Merkle trees. It logs origin, batch, timestamps and ownership changes. [Source pp. 65-66; Example 20]

- Explain node setup, permissions and consensus parameters.
- Define provenance, compliance and routing contract functionality.
- Explain off-chain storage such as IPFS and on-chain Merkle integrity anchoring.
- Detail hybrid consensus, validator selection, fault tolerance and validation. The PDF's example discusses one-third faulty nodes; it is not a universal tolerance guarantee.
- Explain oracle endpoints/integration and IoT sensor protocols such as MQTT.
- The PDF identifies lower-latency finality, reduced on-chain storage, scalability, integrity/authentication and real-time monitoring as possible effects supporting consideration outside 3(k). [Source p. 66; Example 20]

**Plain meaning:** “Put our supply chain on blockchain” does not describe the invention. Teach the actual consensus, storage, verification and sensor integration that create the technical improvements. [Source pp. 64-66]

## 14. Application to present-day software and LLM designs

**This section is a derived reading aid, not additional Patent Office guidance.** The introduction mentions NLP and LLMs, and later sections mention transformers, but the supplied PDF does not give dedicated tests for prompting, RAG, agent orchestration, emotional memory or particular modern product frameworks. Apply its general tests without inventing an automatic answer for those labels. [Source pp. 3, 49-50; §§1.1, 5.2]

- **Prompt or workflow design:** Identify whether the contribution is instructions, a mental/business process or information organization, or an enabled technical solution. A useful answer produced by an ordinary workflow does not itself prove the required technical effect. *Illustration:* an assistant that rearranges recipe advice resembles the information-management concern in example 7. [Derived illustration; source pp. 22-23, 31-37, 42-43]
- **RAG/search/memory:** Distinguish ordinary lookup/personalization from implemented improvements in indexing, storage access, memory utilization or retrieval performance. *Illustration:* a disclosed retrieval architecture that materially improves system performance is the type of contribution Table 1 says to investigate. [Derived illustration; source pp. 39-43]
- **Fine-tuning or dataset selection:** Identify whether the claimed contribution lies in model implementation, essential dataset traits or data curation. *Illustration:* claiming a technical data-harmonization pipeline calls for its sequence, rationale and validation, as in example 16. [Derived illustration; source pp. 59-61]
- **Agents and tool control:** Explain environment interaction, observations, actions and feedback when reinforcement learning is involved; explain actual control interfaces for physical/network systems. *Illustration:* an agent controlling signals must teach how decisions reach the actuators, as in example 12. [Derived illustration; source pp. 37-40, 57-58]
- **Architecture claims:** Explain connections, operational steps and the specific technical effect. *Illustration:* a block diagram containing “memory,” “reasoning” and “learning” boxes is insufficient if it does not teach the claimed mechanism. [Derived illustration; source pp. 21-23, 50-53]

No dedicated framework label changes the need for novelty, non-obviousness, industrial applicability, enablement and applicable exclusion checks. [Source pp. 16-23, 49-50]

## 15. A practical worksheet derived from the document

Use this to collect facts before an invention-specific assessment. It is a study aid, not a form mandated by the PDF.

1. **Define the invention:** What precise product/process and combination of features are you seeking to protect? Which features are essential? [Source pp. 16, 22-23, 35]
2. **State the problem and solution:** Is the core technical, mathematical, commercial, administrative, artistic, informational or mental? Describe the implemented solution, not merely the desired result. [Source pp. 9, 21, 23-35]
3. **Explain the technical effect:** How does the implementation change computing, networking, signal processing or an external technical process beyond ordinary program execution? What results support that explanation? [Source pp. 35, 39-41, 53-61]
4. **Check prior art and obviousness separately:** What was disclosed before the relevant date? What is materially different in the whole claim, and why was that difference not obvious then? [Source pp. 16-19]
5. **Document a workable use:** Identify the concrete industrial use and benefit rather than a speculative application. [Source p. 20]
6. **Write the building instructions:** Components, connections, steps, intermediate states, drawings/flowcharts and the best method known to you. Can the skilled reader reproduce the full claimed scope? [Source pp. 20-23, 51-53]
7. **For AI, disclose the inventive core:** Model/training, transformation, preprocessing, reinforcement-learning interaction, internal-system integration, essential dataset traits or data pipeline, as relevant. Include the stated validation/comparisons. [Source pp. 53-61]
8. **Check inventorship:** Determine the human inventive contribution in light of the PDF's distinction between AI-generated and AI-assisted inventions. [Source p. 50]
9. **Align claims and description:** Clear, succinct, supported scope; unity of invention; structure for any “means for” elements. Include supported method/system forms as appropriate. [Source pp. 22-23]
10. **Apply every relevant exclusion:** Do not assume passing the program-per-se test clears business methods, mathematics, algorithms or other section 3 issues. [Source pp. 9-10, 23-35]

## 16. Final clauses and source-coverage record

Section 6 says Manual chapter **09.03.05.10**, concerning section 3(k), stands deleted when these guidelines come into force. Section 7 says the guidelines apply with immediate effect. The supplied PDF does not give a separate commencement calendar date in that final clause; its cover identifies it as 2025. Do not infer a legal effective date from file metadata. [Source pp. 1, 67; §§6-7]

**Coverage record:** The source was checked from start to end, including image-only flowcharts. The following map lets you return to every substantive part.

- **Source pp. 1-2:** Cover/identity and contents - opening notes and this map.
- **Source pp. 3-5, §1:** Technology context, CRI scope, purpose, legal hierarchy and revisability - guide sections 1-2.
- **Source pp. 5-8, §2:** All 13 definitions - guide section 3.
- **Source pp. 9-10, §§3.1-3.4:** Statutory criteria, 3(k) and 3(l)-(o), legislative intent - guide section 4.
- **Source pp. 10-15, §3.5:** All nine principal CRI case discussions - guide section 4.2.
- **Source pp. 16-17, §4.1:** Seven novelty steps and Manual reference - guide section 5.1.
- **Source pp. 17-19, §4.2:** Inventive-step criteria, authorities and five-step assessment - guide section 5.2.
- **Source p. 20, §4.3:** Industrial applicability - guide section 5.3.
- **Source pp. 20-23, §4.4:** Disclosure, best method, claims, form/substance and means-plus-function - guide section 6.
- **Source pp. 23-26, §4.5.1:** Math test and examples 1-2 - guide section 7.1.
- **Source pp. 26-31, §4.5.2:** Business test, additional contrasts and examples 3-4 - guide section 7.2.
- **Source pp. 31-34, §4.5.3:** Algorithm test and examples 5-6 - guide section 7.3.
- **Source pp. 34-39, §4.5.4:** Program-per-se test and examples 7-8 - guide section 7.4.
- **Source pp. 39-43, Tables 1-2:** Every positive/negative category - guide section 8.
- **Source pp. 43-48, §4.5.5:** Four flowcharts - guide section 9.
- **Source pp. 49-50, §§5.1-5.2.1:** Emerging-technology principles, named model families and AI inventorship - guide section 10.
- **Source pp. 50-53, §5.2.2:** Motupalli disclosure reasoning and supporting authorities - guide section 10.2.
- **Source pp. 53-61, §5.2.3:** All seven scenarios, including 7(a)/7(b), and examples 9-16 - guide section 11.
- **Source pp. 61-64, §5.3:** Quantum principles/components and examples 17-18 - guide section 12.
- **Source pp. 64-66, §5.4:** Blockchain disclosure, smart-contract qualification and examples 19-20 - guide section 13.
- **Source p. 67, §§6-7:** Manual saving clause and applicability - guide section 16.

**Missing from the attachment:** Annexure I (additional illustrative claims/examples) and Annexure II (non-exhaustive case list) are referred to but absent. Filing forms, fees, deadlines, territorial filing strategy, a complete explanation of every statutory exclusion and an invention-specific patentability opinion are not supplied by this document and have not been invented here. [Source pp. 10, 22, 67]

**Separate provenance check:** IP India's official guidelines index lists the 2025 CRI guidelines and Annexures I and II as separate downloads. This contextual check does not add the absent annexures' contents or later legal developments to this summary. [Official index](https://ipindia.gov.in/resource/patents-resources-guidelines), checked 14 October 2026.

**Reading risk:** A favorable hypothetical example is not a grant decision. This guide explains the supplied text; a conclusion about an actual design requires its claims, implementation, human inventorship and prior art to be examined. [Source pp. 16-23, 36, 50, 53]
