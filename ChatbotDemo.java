import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.Duration;
import java.time.format.DateTimeFormatter;
import java.util.stream.Collectors;
import java.util.Objects;

/**
 * Advanced AI Chatbot Backend with NLP and Machine Learning
 * Features: Intent Recognition, Entity Extraction, Context Management, Learning from Conversations
 */
class ChatbotBackend {
    private NLPEngine nlpEngine;
    private MachineLearningModule mlModule;
    private KnowledgeBase knowledgeBase;
    private ConversationManager conversationManager;
    private TrainingModule trainingModule;
    
    public ChatbotBackend() {
        this.nlpEngine = new NLPEngine();
        this.mlModule = new MachineLearningModule();
        this.knowledgeBase = new KnowledgeBase();
        this.conversationManager = new ConversationManager();
        this.trainingModule = new TrainingModule();
        
        // Initialize the system
        initialize();
    }
    
    private void initialize() {
        // Load pre-trained models and knowledge base
        knowledgeBase.loadFAQs();
        mlModule.loadTrainedModels();
        System.out.println("Chatbot Backend initialized successfully!");
    }
    
    /**
     * Main method to process user input and generate response
     */
    public ChatResponse processMessage(String userInput, String sessionId) {
        try {
            // Step 1: Preprocess the input
            String preprocessedInput = nlpEngine.preprocess(userInput);
            
            // Step 2: Extract intents and entities
            IntentResult intentResult = nlpEngine.extractIntent(preprocessedInput);
            List<Entity> entities = nlpEngine.extractEntities(preprocessedInput);
            
            // Step 3: Get conversation context
            ConversationContext context = conversationManager.getContext(sessionId);
            
            // Step 4: Generate response using ML and rule-based approaches
            String response = generateResponse(intentResult, entities, context, preprocessedInput);
            
            // Step 5: Update conversation context
            conversationManager.updateContext(sessionId, userInput, response, intentResult);
            
            // Step 6: Learn from this interaction
            mlModule.learnFromInteraction(userInput, response, intentResult, entities);
            
            return new ChatResponse(response, intentResult.getIntent(), 
                                  intentResult.getConfidence(), entities);
                                  
        } catch (Exception e) {
            System.err.println("Error processing message: " + e.getMessage());
            return new ChatResponse("I'm sorry, I encountered an error. Please try again.", 
                                  "error", 0.0, new ArrayList<>());
        }
    }
    
    private String generateResponse(IntentResult intentResult, List<Entity> entities, 
                                  ConversationContext context, String input) {
        String intent = intentResult.getIntent();
        double confidence = intentResult.getConfidence();
        
        // High confidence responses
        if (confidence > 0.8) {
            return handleHighConfidenceIntent(intent, entities, context);
        }
        
        // Medium confidence - use ML approach
        if (confidence > 0.5) {
            String mlResponse = mlModule.generateMLResponse(input, intent, entities);
            if (mlResponse != null) {
                return mlResponse;
            }
        }
        
        // Low confidence - use rule-based fallback
        return handleLowConfidenceInput(input, context);
    }
    
    private String handleHighConfidenceIntent(String intent, List<Entity> entities, 
                                            ConversationContext context) {
        switch (intent) {
            case "greeting":
                return handleGreeting(context);
            case "question":
                return handleQuestion(entities, context);
            case "technical_help":
                return handleTechnicalHelp(entities);
            case "farewell":
                return handleFarewell(context);
            case "information_request":
                return handleInformationRequest(entities);
            case "complaint":
                return handleComplaint(entities);
            case "praise":
                return handlePraise();
            default:
                return knowledgeBase.getResponse(intent);
        }
    }
    
    private String handleGreeting(ConversationContext context) {
        if (context.isFirstInteraction()) {
            return "Hello! I'm your AI assistant. How can I help you today?";
        } else {
            return "Hi again! What else can I help you with?";
        }
    }
    
    private String handleQuestion(List<Entity> entities, ConversationContext context) {
        for (Entity entity : entities) {
            if (entity.getType().equals("TOPIC")) {
                String response = knowledgeBase.getTopicResponse(entity.getValue());
                if (response != null) {
                    return response;
                }
            }
        }
        return "That's an interesting question! Let me think about that...";
    }
    
    private String handleTechnicalHelp(List<Entity> entities) {
        StringBuilder response = new StringBuilder("I can help you with technical questions. ");
        
        for (Entity entity : entities) {
            if (entity.getType().equals("TECHNOLOGY")) {
                String techResponse = knowledgeBase.getTechnicalResponse(entity.getValue());
                if (techResponse != null) {
                    response.append(techResponse);
                }
            }
        }
        
        return response.toString();
    }
    
    private String handleFarewell(ConversationContext context) {
        context.setEnded(true);
        return "Goodbye! It was nice talking with you. Feel free to reach out anytime!";
    }
    
    private String handleInformationRequest(List<Entity> entities) {
        return "I'll help you find that information. What specifically would you like to know?";
    }
    
    private String handleComplaint(List<Entity> entities) {
        return "I understand your concern. Let me help you resolve this issue. Could you provide more details?";
    }
    
    private String handlePraise() {
        return "Thank you for the kind words! I'm here to help whenever you need assistance.";
    }
    
    private String handleLowConfidenceInput(String input, ConversationContext context) {
        // Try pattern matching
        String patternResponse = nlpEngine.matchPatterns(input);
        if (patternResponse != null) {
            return patternResponse;
        }
        
        // Try similarity matching with knowledge base
        String similarResponse = knowledgeBase.findSimilarResponse(input);
        if (similarResponse != null) {
            return similarResponse;
        }
        
        // Default response
        return "I'm not entirely sure about that. Could you rephrase your question or ask something else?";
    }
    
    /**
     * Train the bot with FAQ data
     */
    public void trainWithFAQs(List<FAQItem> faqItems) {
        trainingModule.trainWithFAQs(faqItems);
        mlModule.updateModel(faqItems);
        System.out.println("Training completed with " + faqItems.size() + " FAQ items.");
    }
    
    /**
     * Add new knowledge to the bot
     */
    public void addKnowledge(String question, String answer, String category) {
        knowledgeBase.addKnowledge(question, answer, category);
        mlModule.incrementalLearning(question, answer, category);
    }
    
    /**
     * Get bot statistics
     */
    public BotStatistics getStatistics() {
        return new BotStatistics(
            conversationManager.getTotalConversations(),
            mlModule.getModelAccuracy(),
            knowledgeBase.getKnowledgeCount(),
            conversationManager.getAverageConversationLength()
        );
    }
}

/**
 * Advanced NLP Engine with intent recognition and entity extraction
 */
class NLPEngine {
    private IntentClassifier intentClassifier;
    private EntityExtractor entityExtractor;
    private TextPreprocessor preprocessor;
    private PatternMatcher patternMatcher;
    
    public NLPEngine() {
        this.intentClassifier = new IntentClassifier();
        this.entityExtractor = new EntityExtractor();
        this.preprocessor = new TextPreprocessor();
        this.patternMatcher = new PatternMatcher();
    }
    
    public String preprocess(String text) {
        return preprocessor.process(text);
    }
    
    public IntentResult extractIntent(String text) {
        return intentClassifier.classify(text);
    }
    
    public List<Entity> extractEntities(String text) {
        return entityExtractor.extract(text);
    }
    
    public String matchPatterns(String text) {
        return patternMatcher.match(text);
    }
}

/**
 * Intent Classification using rule-based and ML approaches
 */
class IntentClassifier {
    private Map<String, List<String>> intentPatterns;
    private Map<String, Double> intentWeights;
    
    public IntentClassifier() {
        initializeIntentPatterns();
    }
    
    private void initializeIntentPatterns() {
        intentPatterns = new HashMap<>();
        intentWeights = new HashMap<>();
        
        // Greeting patterns
        intentPatterns.put("greeting", Arrays.asList(
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"
        ));
        
        // Question patterns
        intentPatterns.put("question", Arrays.asList(
            "what", "how", "why", "when", "where", "which", "can you", "do you know"
        ));
        
        // Technical help patterns
        intentPatterns.put("technical_help", Arrays.asList(
            "help", "problem", "issue", "error", "bug", "not working", "technical", "support"
        ));
        
        // Farewell patterns
        intentPatterns.put("farewell", Arrays.asList(
            "bye", "goodbye", "see you", "farewell", "take care", "until next time"
        ));
        
        // Information request patterns
        intentPatterns.put("information_request", Arrays.asList(
            "tell me about", "information about", "details about", "explain", "describe"
        ));
        
        // Complaint patterns
        intentPatterns.put("complaint", Arrays.asList(
            "complain", "complaint", "problem with", "issue with", "not satisfied", "disappointed"
        ));
        
        // Praise patterns
        intentPatterns.put("praise", Arrays.asList(
            "thank you", "thanks", "great job", "excellent", "amazing", "wonderful", "helpful"
        ));
        
        // Set weights for each intent
        intentWeights.put("greeting", 1.0);
        intentWeights.put("question", 1.2);
        intentWeights.put("technical_help", 1.5);
        intentWeights.put("farewell", 1.0);
        intentWeights.put("information_request", 1.3);
        intentWeights.put("complaint", 1.4);
        intentWeights.put("praise", 1.1);
    }
    
    public IntentResult classify(String text) {
        String lowerText = text.toLowerCase();
        Map<String, Double> intentScores = new HashMap<>();
        
        for (String intent : intentPatterns.keySet()) {
            double score = 0.0;
            List<String> patterns = intentPatterns.get(intent);
            
            for (String pattern : patterns) {
                if (lowerText.contains(pattern)) {
                    score += intentWeights.get(intent);
                    
                    // Boost score for exact matches
                    if (lowerText.equals(pattern)) {
                        score += 0.5;
                    }
                    
                    // Boost score for pattern at beginning
                    if (lowerText.startsWith(pattern)) {
                        score += 0.3;
                    }
                }
            }
            
            if (score > 0) {
                intentScores.put(intent, score);
            }
        }
        
        if (intentScores.isEmpty()) {
            return new IntentResult("unknown", 0.0);
        }
        
        // Find the intent with highest score
        String bestIntent = intentScores.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .get().getKey();
        
        double confidence = Math.min(intentScores.get(bestIntent) / 3.0, 1.0);
        
        return new IntentResult(bestIntent, confidence);
    }
}

/**
 * Entity Extraction for identifying important information in text
 */
class EntityExtractor {
    private Map<String, Pattern> entityPatterns;
    
    public EntityExtractor() {
        initializeEntityPatterns();
    }
    
    private void initializeEntityPatterns() {
        entityPatterns = new HashMap<>();
        
        // Technology entities
        entityPatterns.put("TECHNOLOGY", Pattern.compile(
            "\\b(java|python|javascript|html|css|sql|database|programming|coding|software|ai|machine learning|nlp)\\b",
            Pattern.CASE_INSENSITIVE
        ));
        
        // Topic entities
        entityPatterns.put("TOPIC", Pattern.compile(
            "\\b(weather|time|date|news|sports|music|movies|books|science|history|geography)\\b",
            Pattern.CASE_INSENSITIVE
        ));
        
        // Number entities
        entityPatterns.put("NUMBER", Pattern.compile("\\b\\d+\\b"));
        
        // Email entities
        entityPatterns.put("EMAIL", Pattern.compile(
            "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"
        ));
        
        // URL entities
        entityPatterns.put("URL", Pattern.compile(
            "https?://[\\w\\-]+(\\.[\\w\\-]+)+([\\w\\-\\.,@?^=%&:/~\\+#]*[\\w\\-\\@?^=%&/~\\+#])?"
        ));
        
        // Date entities
        entityPatterns.put("DATE", Pattern.compile(
            "\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b|\\b(today|tomorrow|yesterday)\\b",
            Pattern.CASE_INSENSITIVE
        ));
        
        // Time entities
        entityPatterns.put("TIME", Pattern.compile(
            "\\b\\d{1,2}:\\d{2}\\s?(AM|PM|am|pm)?\\b|\\b(morning|afternoon|evening|night)\\b",
            Pattern.CASE_INSENSITIVE
        ));
    }
    
    public List<Entity> extract(String text) {
        List<Entity> entities = new ArrayList<>();
        
        for (String entityType : entityPatterns.keySet()) {
            Pattern pattern = entityPatterns.get(entityType);
            Matcher matcher = pattern.matcher(text);
            
            while (matcher.find()) {
                String value = matcher.group().trim();
                int start = matcher.start();
                int end = matcher.end();
                
                entities.add(new Entity(entityType, value, start, end));
            }
        }
        
        return entities;
    }
}

/**
 * Text preprocessing for cleaning and normalizing input
 */
class TextPreprocessor {
    private Set<String> stopWords;
    private Pattern punctuationPattern;
    
    public TextPreprocessor() {
        initializeStopWords();
        punctuationPattern = Pattern.compile("[^a-zA-Z0-9\\s]");
    }
    
    private void initializeStopWords() {
        stopWords = new HashSet<>(Arrays.asList(
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with", "would",
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
            "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"
        ));
    }
    
    public String process(String text) {
        // Convert to lowercase
        text = text.toLowerCase();
        
        // Remove extra whitespace
        text = text.trim().replaceAll("\\s+", " ");
        
        // Handle contractions
        text = expandContractions(text);
        
        return text;
    }
    
    private String expandContractions(String text) {
        Map<String, String> contractions = new HashMap<>();
        contractions.put("don't", "do not");
        contractions.put("won't", "will not");
        contractions.put("can't", "cannot");
        contractions.put("i'm", "i am");
        contractions.put("you're", "you are");
        contractions.put("it's", "it is");
        contractions.put("we're", "we are");
        contractions.put("they're", "they are");
        contractions.put("isn't", "is not");
        contractions.put("aren't", "are not");
        contractions.put("wasn't", "was not");
        contractions.put("weren't", "were not");
        contractions.put("haven't", "have not");
        contractions.put("hasn't", "has not");
        contractions.put("hadn't", "had not");
        contractions.put("doesn't", "does not");
        contractions.put("didn't", "did not");
        
        for (Map.Entry<String, String> entry : contractions.entrySet()) {
            text = text.replace(entry.getKey(), entry.getValue());
        }
        
        return text;
    }
}

/**
 * Pattern matching for common queries
 */
class PatternMatcher {
    private Map<Pattern, String> patterns;
    
    public PatternMatcher() {
        initializePatterns();
    }
    
    private void initializePatterns() {
        patterns = new HashMap<>();
        
        // Time patterns
        patterns.put(Pattern.compile("what.*(time|clock)", Pattern.CASE_INSENSITIVE),
                    "The current time is " + LocalDateTime.now().format(DateTimeFormatter.ofPattern("HH:mm")));
        
        // Date patterns
        patterns.put(Pattern.compile("what.*(date|today)", Pattern.CASE_INSENSITIVE),
                    "Today is " + LocalDateTime.now().format(DateTimeFormatter.ofPattern("MMMM dd, yyyy")));
        
        // Math patterns
        patterns.put(Pattern.compile("\\d+\\s*[+\\-*/]\\s*\\d+", Pattern.CASE_INSENSITIVE),
                    "I can see you're asking about math! For calculations, I'd recommend using a calculator.");
        
        // Weather patterns
        patterns.put(Pattern.compile("weather|temperature|rain|sunny|cloudy", Pattern.CASE_INSENSITIVE),
                    "I don't have access to real-time weather data. Please check a weather app or website.");
        
        // Name patterns
        patterns.put(Pattern.compile("what.*(your name|are you called)", Pattern.CASE_INSENSITIVE),
                    "I'm an AI assistant created to help answer your questions and have conversations!");
        
        // Capability patterns
        patterns.put(Pattern.compile("what.*can.*you.*do", Pattern.CASE_INSENSITIVE),
                    "I can help with questions, provide information, assist with programming concepts, and have conversations!");
    }
    
    public String match(String text) {
        for (Map.Entry<Pattern, String> entry : patterns.entrySet()) {
            if (entry.getKey().matcher(text).find()) {
                return entry.getValue();
            }
        }
        return null;
    }
}

/**
 * Machine Learning Module for learning from conversations
 */
class MachineLearningModule {
    private Map<String, List<String>> intentExamples;
    private Map<String, Double> responseScores;
    private List<ConversationLog> trainingData;
    private double modelAccuracy;
    
    public MachineLearningModule() {
        this.intentExamples = new ConcurrentHashMap<>();
        this.responseScores = new ConcurrentHashMap<>();
        this.trainingData = new ArrayList<>();
        this.modelAccuracy = 0.75; // Initial accuracy
    }
    
    public void loadTrainedModels() {
        // Load pre-trained models from files
        System.out.println("Loading trained models...");
        // Implementation would load from files or database
    }
    
    public String generateMLResponse(String input, String intent, List<Entity> entities) {
        // Use machine learning to generate contextual responses
        List<String> examples = intentExamples.get(intent);
        if (examples != null && !examples.isEmpty()) {
            // Find most similar example and return associated response
            String mostSimilar = findMostSimilarExample(input, examples);
            return generateResponseFromExample(mostSimilar, entities);
        }
        return null;
    }
    
    private String findMostSimilarExample(String input, List<String> examples) {
        double maxSimilarity = 0.0;
        String mostSimilar = examples.get(0);
        
        for (String example : examples) {
            double similarity = calculateSimilarity(input, example);
            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                mostSimilar = example;
            }
        }
        
        return mostSimilar;
    }
    
    private double calculateSimilarity(String text1, String text2) {
        Set<String> words1 = new HashSet<>(Arrays.asList(text1.toLowerCase().split("\\s+")));
        Set<String> words2 = new HashSet<>(Arrays.asList(text2.toLowerCase().split("\\s+")));
        
        Set<String> intersection = new HashSet<>(words1);
        intersection.retainAll(words2);
        
        Set<String> union = new HashSet<>(words1);
        union.addAll(words2);
        
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }
    
    private String generateResponseFromExample(String example, List<Entity> entities) {
        // Generate response based on example and entities
        return "Based on similar queries, here's what I can help you with...";
    }
    
    public void learnFromInteraction(String input, String response, IntentResult intent, List<Entity> entities) {
        // Store interaction for learning
        ConversationLog log = new ConversationLog(input, response, intent.getIntent(), entities);
        trainingData.add(log);
        
        // Update intent examples
        intentExamples.computeIfAbsent(intent.getIntent(), k -> new ArrayList<>()).add(input);
        
        // Update model accuracy based on interaction success
        updateModelAccuracy(intent.getConfidence());
    }
    
    private void updateModelAccuracy(double confidence) {
        // Simple accuracy update based on confidence
        modelAccuracy = (modelAccuracy * 0.9) + (confidence * 0.1);
    }
    
    public void updateModel(List<FAQItem> faqItems) {
        // Update ML model with new FAQ data
        for (FAQItem faq : faqItems) {
            intentExamples.computeIfAbsent(faq.getCategory(), k -> new ArrayList<>()).add(faq.getQuestion());
        }
        System.out.println("Model updated with " + faqItems.size() + " FAQ items");
    }
    
    public void incrementalLearning(String question, String answer, String category) {
        // Implement incremental learning
        intentExamples.computeIfAbsent(category, k -> new ArrayList<>()).add(question);
        System.out.println("Incremental learning: Added new knowledge for category " + category);
    }
    
    public double getModelAccuracy() {
        return modelAccuracy;
    }
}

/**
 * Knowledge Base for storing and retrieving information
 */
class KnowledgeBase {
    private Map<String, String> responses;
    private Map<String, List<String>> categoryResponses;
    private Map<String, String> technicalResponses;
    private List<FAQItem> faqItems;
    
    public KnowledgeBase() {
        this.responses = new ConcurrentHashMap<>();
        this.categoryResponses = new ConcurrentHashMap<>();
        this.technicalResponses = new ConcurrentHashMap<>();
        this.faqItems = new ArrayList<>();
        initializeKnowledgeBase();
    }
    
    private void initializeKnowledgeBase() {
        // Programming and Technical responses
        technicalResponses.put("java", "Java is a versatile, object-oriented programming language. It's platform-independent and widely used for enterprise applications, Android development, and web services.");
        technicalResponses.put("python", "Python is a high-level programming language known for its simplicity and readability. It's popular for data science, AI, web development, and automation.");
        technicalResponses.put("javascript", "JavaScript is a dynamic programming language primarily used for web development. It can run in browsers and servers (Node.js).");
        technicalResponses.put("html", "HTML (HyperText Markup Language) is the standard markup language for creating web pages and web applications.");
        technicalResponses.put("css", "CSS (Cascading Style Sheets) is used for describing the presentation of HTML documents, including layout, colors, and fonts.");
        technicalResponses.put("sql", "SQL (Structured Query Language) is used for managing and querying relational databases.");
        technicalResponses.put("ai", "Artificial Intelligence refers to computer systems that can perform tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.");
        technicalResponses.put("machine learning", "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.");
        technicalResponses.put("nlp", "Natural Language Processing is a branch of AI that helps computers understand, interpret, and generate human language.");
        
        // General responses
        responses.put("greeting", "Hello! I'm here to help you with any questions you might have.");
        responses.put("farewell", "Goodbye! Feel free to ask me anything anytime.");
        responses.put("unknown", "I'm not sure about that. Could you please rephrase your question?");
        responses.put("praise", "Thank you! I'm glad I could help you.");
        responses.put("complaint", "I apologize for any inconvenience. How can I help resolve this issue?");
    }
    
    public void loadFAQs() {
        // Load FAQ items from external source
        System.out.println("Loading FAQ database...");
        // Implementation would load from files or database
    }
    
    public String getResponse(String intent) {
        return responses.get(intent);
    }
    
    public String getTopicResponse(String topic) {
        return technicalResponses.get(topic.toLowerCase());
    }
    
    public String getTechnicalResponse(String technology) {
        return technicalResponses.get(technology.toLowerCase());
    }
    
    public String findSimilarResponse(String input) {
        // Find similar responses using text similarity
        double maxSimilarity = 0.0;
        String bestResponse = null;
        
        for (Map.Entry<String, String> entry : technicalResponses.entrySet()) {
            double similarity = calculateSimilarity(input, entry.getKey());
            if (similarity > maxSimilarity && similarity > 0.3) {
                maxSimilarity = similarity;
                bestResponse = entry.getValue();
            }
        }
        
        return bestResponse;
    }
    
    private double calculateSimilarity(String text1, String text2) {
        Set<String> words1 = new HashSet<>(Arrays.asList(text1.toLowerCase().split("\\s+")));
        Set<String> words2 = new HashSet<>(Arrays.asList(text2.toLowerCase().split("\\s+")));
        
        Set<String> intersection = new HashSet<>(words1);
        intersection.retainAll(words2);
        
        Set<String> union = new HashSet<>(words1);
        union.addAll(words2);
        
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }
    
    public void addKnowledge(String question, String answer, String category) {
        responses.put(question.toLowerCase(), answer);
        categoryResponses.computeIfAbsent(category, k -> new ArrayList<>()).add(answer);
        System.out.println("Added new knowledge: " + question);
    }
    
    public int getKnowledgeCount() {
        return responses.size() + technicalResponses.size();
    }
}

/**
 * Conversation Management for context tracking
 */
class ConversationManager {
    private Map<String, ConversationContext> activeContexts;
    private Map<String, List<String>> conversationHistory;
    private int totalConversations;
    
    public ConversationManager() {
        this.activeContexts = new ConcurrentHashMap<>();
        this.conversationHistory = new ConcurrentHashMap<>();
        this.totalConversations = 0;
    }
    
    public ConversationContext getContext(String sessionId) {
        return activeContexts.computeIfAbsent(sessionId, k -> new ConversationContext(sessionId));
    }
    
    public void updateContext(String sessionId, String userInput, String botResponse, IntentResult intent) {
        ConversationContext context = getContext(sessionId);
        context.addInteraction(userInput, botResponse, intent);
        
        // Update conversation history
        conversationHistory.computeIfAbsent(sessionId, k -> new ArrayList<>())
                           .add("User: " + userInput + " | Bot: " + botResponse);
        
        if (context.isFirstInteraction()) {
            totalConversations++;
        }
    }
    
    public int getTotalConversations() {
        return totalConversations;
    }
    
    public double getAverageConversationLength() {
        if (conversationHistory.isEmpty()) return 0.0;
        
        double totalLength = conversationHistory.values().stream()
                                               .mapToInt(List::size)
                                               .sum();
        return totalLength / conversationHistory.size();
    }
}

/**
 * Training Module for FAQ and conversation data
 */
class TrainingModule {
    private List<FAQItem> trainingFAQs;
    private Map<String, Integer> categoryCount;
    
    public TrainingModule() {
        this.trainingFAQs = new ArrayList<>();
        this.categoryCount = new HashMap<>();
    }
    
    public void trainWithFAQs(List<FAQItem> faqItems) {
this.trainingFAQs.addAll(faqItems);
        
        // Count categories for training statistics
        for (FAQItem faq : faqItems) {
            categoryCount.put(faq.getCategory(), 
                            categoryCount.getOrDefault(faq.getCategory(), 0) + 1);
        }
        
        // Process training data
        processTrainingData();
        
        System.out.println("Training completed successfully!");
        System.out.println("Total FAQ items: " + faqItems.size());
        System.out.println("Categories trained: " + categoryCount.size());
    }
    
    private void processTrainingData() {
        // Process the training data for better model performance
        for (FAQItem faq : trainingFAQs) {
            // Validate and clean the FAQ data
            if (faq.getQuestion() != null && faq.getAnswer() != null) {
                faq.setQuestion(faq.getQuestion().trim());
                faq.setAnswer(faq.getAnswer().trim());
            }
        }
    }
    
    public List<FAQItem> getTrainingFAQs() {
        return new ArrayList<>(trainingFAQs);
    }
    
    public Map<String, Integer> getCategoryStatistics() {
        return new HashMap<>(categoryCount);
    }
}

/**
 * Data classes for storing conversation and bot information
 */

class ChatResponse {
    private String message;
    private String intent;
    private double confidence;
    private List<Entity> entities;
    private LocalDateTime timestamp;
    
    public ChatResponse(String message, String intent, double confidence, List<Entity> entities) {
        this.message = message;
        this.intent = intent;
        this.confidence = confidence;
        this.entities = entities;
        this.timestamp = LocalDateTime.now();
    }
    
    // Getters
    public String getMessage() { return message; }
    public String getIntent() { return intent; }
    public double getConfidence() { return confidence; }
    public List<Entity> getEntities() { return entities; }
    public LocalDateTime getTimestamp() { return timestamp; }
    
    @Override
    public String toString() {
        return String.format("ChatResponse{message='%s', intent='%s', confidence=%.2f, entities=%d}", 
                           message, intent, confidence, entities.size());
    }
}

class IntentResult {
    private String intent;
    private double confidence;
    
    public IntentResult(String intent, double confidence) {
        this.intent = intent;
        this.confidence = confidence;
    }
    
    public String getIntent() { return intent; }
    public double getConfidence() { return confidence; }
    
    @Override
    public String toString() {
        return String.format("IntentResult{intent='%s', confidence=%.2f}", intent, confidence);
    }
}

class Entity {
    private String type;
    private String value;
    private int startPosition;
    private int endPosition;
    
    public Entity(String type, String value, int startPosition, int endPosition) {
        this.type = type;
        this.value = value;
        this.startPosition = startPosition;
        this.endPosition = endPosition;
    }
    
    // Getters
    public String getType() { return type; }
    public String getValue() { return value; }
    public int getStartPosition() { return startPosition; }
    public int getEndPosition() { return endPosition; }
    
    @Override
    public String toString() {
        return String.format("Entity{type='%s', value='%s', pos=[%d-%d]}", 
                           type, value, startPosition, endPosition);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Entity entity = (Entity) obj;
        return startPosition == entity.startPosition && 
               endPosition == entity.endPosition &&
               Objects.equals(type, entity.type) && 
               Objects.equals(value, entity.value);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(type, value, startPosition, endPosition);
    }
}

class ConversationContext {
    private String sessionId;
    private List<String> conversationHistory;
    private Map<String, Object> contextData;
    private LocalDateTime startTime;
    private LocalDateTime lastInteraction;
    private boolean isEnded;
    private int interactionCount;
    
    public ConversationContext(String sessionId) {
        this.sessionId = sessionId;
        this.conversationHistory = new ArrayList<>();
        this.contextData = new HashMap<>();
        this.startTime = LocalDateTime.now();
        this.lastInteraction = LocalDateTime.now();
        this.isEnded = false;
        this.interactionCount = 0;
    }
    
    public void addInteraction(String userInput, String botResponse, IntentResult intent) {
        conversationHistory.add("User: " + userInput);
        conversationHistory.add("Bot: " + botResponse);
        lastInteraction = LocalDateTime.now();
        interactionCount++;
        
        // Store context data
        contextData.put("lastIntent", intent.getIntent());
        contextData.put("lastUserInput", userInput);
        contextData.put("lastBotResponse", botResponse);
    }
    
    public boolean isFirstInteraction() {
        return interactionCount <= 1;
    }
    
    public Duration getConversationDuration() {
        return Duration.between(startTime, lastInteraction);
    }
    
    // Getters and setters
    public String getSessionId() { return sessionId; }
    public List<String> getConversationHistory() { return new ArrayList<>(conversationHistory); }
    public Map<String, Object> getContextData() { return new HashMap<>(contextData); }
    public LocalDateTime getStartTime() { return startTime; }
    public LocalDateTime getLastInteraction() { return lastInteraction; }
    public boolean isEnded() { return isEnded; }
    public void setEnded(boolean ended) { isEnded = ended; }
    public int getInteractionCount() { return interactionCount; }
    
    public Object getContextValue(String key) {
        return contextData.get(key);
    }
    
    public void setContextValue(String key, Object value) {
        contextData.put(key, value);
    }
    
    @Override
    public String toString() {
        return String.format("ConversationContext{sessionId='%s', interactions=%d, duration=%s}", 
                           sessionId, interactionCount, getConversationDuration());
    }
}

class FAQItem {
    private String question;
    private String answer;
    private String category;
    private List<String> keywords;
    private double relevanceScore;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public FAQItem(String question, String answer, String category) {
        this.question = question;
        this.answer = answer;
        this.category = category;
        this.keywords = extractKeywords(question);
        this.relevanceScore = 1.0;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    private List<String> extractKeywords(String text) {
        // Simple keyword extraction
        return Arrays.asList(text.toLowerCase().split("\\s+"))
                    .stream()
                    .filter(word -> word.length() > 3)
                    .collect(Collectors.toList());
    }
    
    // Getters and setters
    public String getQuestion() { return question; }
    public void setQuestion(String question) { 
        this.question = question;
        this.keywords = extractKeywords(question);
        this.updatedAt = LocalDateTime.now();
    }
    
    public String getAnswer() { return answer; }
    public void setAnswer(String answer) { 
        this.answer = answer;
        this.updatedAt = LocalDateTime.now();
    }
    
    public String getCategory() { return category; }
    public void setCategory(String category) { 
        this.category = category;
        this.updatedAt = LocalDateTime.now();
    }
    
    public List<String> getKeywords() { return new ArrayList<>(keywords); }
    public double getRelevanceScore() { return relevanceScore; }
    public void setRelevanceScore(double relevanceScore) { this.relevanceScore = relevanceScore; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    
    @Override
    public String toString() {
        return String.format("FAQItem{question='%s', category='%s', score=%.2f}", 
                           question, category, relevanceScore);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        FAQItem faqItem = (FAQItem) obj;
        return Objects.equals(question, faqItem.question) && 
               Objects.equals(category, faqItem.category);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(question, category);
    }
}

class ConversationLog {
    private String userInput;
    private String botResponse;
    private String intent;
    private List<Entity> entities;
    private LocalDateTime timestamp;
    private String sessionId;
    
    public ConversationLog(String userInput, String botResponse, String intent, List<Entity> entities) {
        this.userInput = userInput;
        this.botResponse = botResponse;
        this.intent = intent;
        this.entities = new ArrayList<>(entities);
        this.timestamp = LocalDateTime.now();
    }
    
    // Getters
    public String getUserInput() { return userInput; }
    public String getBotResponse() { return botResponse; }
    public String getIntent() { return intent; }
    public List<Entity> getEntities() { return new ArrayList<>(entities); }
    public LocalDateTime getTimestamp() { return timestamp; }
    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    
    @Override
    public String toString() {
        return String.format("ConversationLog{input='%s', intent='%s', timestamp=%s}", 
                           userInput, intent, timestamp);
    }
}

class BotStatistics {
    private int totalConversations;
    private double modelAccuracy;
    private int knowledgeCount;
    private double averageConversationLength;
    private LocalDateTime lastUpdated;
    
    public BotStatistics(int totalConversations, double modelAccuracy, 
                        int knowledgeCount, double averageConversationLength) {
        this.totalConversations = totalConversations;
        this.modelAccuracy = modelAccuracy;
        this.knowledgeCount = knowledgeCount;
        this.averageConversationLength = averageConversationLength;
        this.lastUpdated = LocalDateTime.now();
    }
    
    // Getters
    public int getTotalConversations() { return totalConversations; }
    public double getModelAccuracy() { return modelAccuracy; }
    public int getKnowledgeCount() { return knowledgeCount; }
    public double getAverageConversationLength() { return averageConversationLength; }
    public LocalDateTime getLastUpdated() { return lastUpdated; }
    
    @Override
    public String toString() {
        return String.format("BotStatistics{conversations=%d, accuracy=%.2f%%, knowledge=%d, avgLength=%.1f}", 
                           totalConversations, modelAccuracy * 100, knowledgeCount, averageConversationLength);
    }
}

/**
 * Main class to demonstrate the chatbot functionality
 */
public class ChatbotDemo {
    public static void main(String[] args) {
        // Initialize the chatbot
        ChatbotBackend chatbot = new ChatbotBackend();
        
        // Add some sample knowledge
        chatbot.addKnowledge("What is Java?", "Java is a programming language", "programming");
        chatbot.addKnowledge("How to learn Python?", "Start with basics and practice regularly", "programming");
        
        // Create sample FAQ items
        List<FAQItem> faqItems = Arrays.asList(
            new FAQItem("What is machine learning?", "Machine learning is a subset of AI", "ai"),
            new FAQItem("How does NLP work?", "NLP processes human language using algorithms", "ai"),
            new FAQItem("What is a database?", "A database is a structured collection of data", "database")
        );
        
        // Train the bot
        chatbot.trainWithFAQs(faqItems);
        
        // Test conversations
        testConversations(chatbot);
        
        // Display statistics
        BotStatistics stats = chatbot.getStatistics();
        System.out.println("\n=== Bot Statistics ===");
        System.out.println(stats);
    }
    
    private static void testConversations(ChatbotBackend chatbot) {
        String sessionId = "test-session-1";
        
        // Test different types of inputs
        String[] testInputs = {
            "Hello there!",
            "What is Java?",
            "Can you help me with programming?",
            "Tell me about machine learning",
            "What time is it?",
            "Thank you for your help",
            "Goodbye!"
        };
        
        System.out.println("\n=== Testing Conversations ===");
        
        for (String input : testInputs) {
            ChatResponse response = chatbot.processMessage(input, sessionId);
            System.out.println("User: " + input);
            System.out.println("Bot: " + response.getMessage());
            System.out.println("Intent: " + response.getIntent() + 
                             " (Confidence: " + String.format("%.2f", response.getConfidence()) + ")");
            
            if (!response.getEntities().isEmpty()) {
                System.out.println("Entities: " + response.getEntities());
            }
            
            System.out.println("---");
        }
    }
}