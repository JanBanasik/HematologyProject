package com.agh.anemia.model;

public class ChatMessageResponse {
    private String reply;

    public ChatMessageResponse(String reply) {
        this.reply = reply;
    }

    public String getReply() {
        return reply;
    }

    public void setReply(String reply) {
        this.reply = reply;
    }
}