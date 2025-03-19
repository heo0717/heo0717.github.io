---
layout: archive
title: "portfolio"
permalink: /portfolio/
author_profile: true
---

{% for post in site.posts %}
  - [{{ post.title }}]({{ post.url }})
{% endfor %}