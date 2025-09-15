FROM confluentinc/cp-kafka-connect:7.2.15

ENV CONNECT_PLUGIN_PATH="/usr/share/java,/usr/share/confluent-hub-components"

RUN  confluent-hub install --no-prompt confluentinc/kafka-connect-datagen:latest

# Install MongoDB Sink Connector
RUN confluent-hub install --no-prompt mongodb/kafka-connect-mongodb:latest