---
layout: post
title: Algorithm and data struct
---

# Algorithm & Datastruct
{% include _shared_block/Concept/Algorithm.md %}
{% include _shared_block/Concept/DataStruct.md %}


{% assign datas = site.data.AlgorithmAndDataStruct %}

{% for data in datas %}

<!-- 목차 생성 -->
{% if data.Type == "Title" %}
{{ data.Header }}# {{ data.Name }}
{% endif %}

<!-- 내용 추가 -->
{% if data.Type == "Content" %}
* [{{data.Name}}]({{data.Path}})
{% endif %}

{% endfor %}
